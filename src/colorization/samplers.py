from tqdm import tqdm

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def sample_stage_1(
    model,
    prompt_embeds,
    negative_prompt_embeds, 
    views,
    height=None,
    width=None,
    fixed_im=None,
    num_inference_steps=100,
    guidance_scale=7.0,
    reduction='mean',
    generator=None,
    num_recurse=1,
    start_diffusion_step=0
):

    # Params
    num_images_per_prompt = 1
    device = torch.device('cuda')   # Sometimes model device is cpu???
    height = model.unet.config.sample_size if height is None else height
    width = model.unet.config.sample_size if width is None else width
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    assert num_prompts == len(views), \
        "Number of prompts must match number of views!"

    # Resize image to correct size
    if fixed_im is not None:
        fixed_im = TF.resize(fixed_im, height, antialias=False)

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # Make intermediate_images
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        model.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    for i, t in enumerate(timesteps):
        for j in range(num_recurse):
            # Logic to keep a component fixed to reference image
            if fixed_im is not None:
                # Inject noise
                alpha_cumprod = model.scheduler.alphas_cumprod[t]
                im_noisy = torch.sqrt(alpha_cumprod) * fixed_im + torch.sqrt(1 - alpha_cumprod) * torch.randn_like(fixed_im)

                # Replace component in noisy images with component from fixed image
                # im_noisy_component = torch.zeros(im_noisy.shape, device=noisy_images.device, dtype=noisy_images.dtype)
                # noisy_images_component = torch.zeros(im_noisy.shape, device=noisy_images.device, dtype=noisy_images.dtype)
                # for i, view in enumerate(views):
                #     if i % 2 == 0:
                #         im_noisy_component += view.imprint(im_noisy)
                #     else:
                #         noisy_images_component += view.imprint(noisy_images[0])
                # noisy_images = im_noisy_component + noisy_images_component

                # # Correct for factor of 2 from view TODO: Fix this....
                # noisy_images = noisy_images[None] / len(views)
                im_noisy_component = views[0].inverse_view(im_noisy).to(noisy_images.device).to(noisy_images.dtype)
                noisy_images_component = views[1].inverse_view(noisy_images[0])
                noisy_images = im_noisy_component + noisy_images_component

                # Correct for factor of 2 from view TODO: Fix this....
                noisy_images = noisy_images[None] / 2.
                
                # "Reset" diffusion by replacing noisy images with noisy version
                # of grayscale image. All diffusion steps before this one are "thrown away"
                if i == start_diffusion_step:
                    noisy_images = im_noisy.to(noisy_images.device).to(noisy_images.dtype)[None]

            # Apply views to noisy_image
            viewed_noisy_images = []
            for view_fn in views:
                viewed_noisy_images.append(view_fn.view(noisy_images[0]))
            viewed_noisy_images = torch.stack(viewed_noisy_images)

            # Duplicate inputs for CFG
            # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
            model_input = torch.cat([viewed_noisy_images] * 2)
            model_input = model.scheduler.scale_model_input(model_input, t)

            # Predict noise estimate
            noise_pred = model.unet(
                model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # Extract uncond (neg) and cond noise estimates
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # Invert the unconditional (negative) estimates
            inverted_preds = []
            for pred, view in zip(noise_pred_uncond, views):
                inverted_pred = view.inverse_view(pred)
                inverted_preds.append(inverted_pred)
            noise_pred_uncond = torch.stack(inverted_preds)

            # Invert the conditional estimates
            inverted_preds = []
            for pred, view in zip(noise_pred_text, views):
                inverted_pred = view.inverse_view(pred)
                inverted_preds.append(inverted_pred)
            noise_pred_text = torch.stack(inverted_preds)

            # Split into noise estimate and variance estimates
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Reduce predicted noise and variances
            noise_pred = noise_pred.view(-1,num_prompts,3,height,width)
            predicted_variance = predicted_variance.view(-1,num_prompts,3,height,width)
            if reduction == 'mean':
                noise_pred = noise_pred.mean(1)
                predicted_variance = predicted_variance.mean(1)
            elif reduction == 'alternate':
                noise_pred = noise_pred[:,i%num_prompts]
                predicted_variance = predicted_variance[:,i%num_prompts]
            else:
                raise ValueError('Reduction must be either `mean` or `alternate`')
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            noisy_images = model.scheduler.step(
                noise_pred, t, noisy_images, generator=generator, return_dict=False
            )[0]

            if j != (num_recurse - 1):
                beta = model.scheduler.betas[t]
                noisy_images = (1 - beta).sqrt() * noisy_images + beta.sqrt() * torch.randn_like(noisy_images)

    # Return denoised images
    return noisy_images





@torch.no_grad()
def sample_stage_2(
    model,
    image,
    prompt_embeds,
    negative_prompt_embeds, 
    views,
    height=None,
    width=None,
    fixed_im=None,
    num_inference_steps=100,
    guidance_scale=7.0,
    reduction='mean',
    noise_level=50,
    generator=None
):

    # Params
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    height = model.unet.config.sample_size if height is None else height
    width = model.unet.config.sample_size if width is None else width
    device = image.device
    num_images_per_prompt = 1
    
    # Resize fixed image to correct size
    if fixed_im is not None:
        fixed_im = TF.resize(fixed_im, height, antialias=False)

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Get timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    num_channels = model.unet.config.in_channels // 2
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        num_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # Prepare upscaled image and noise level
    image = model.preprocess_image(image, num_images_per_prompt, device)
    upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

    noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
    noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
    upscaled = model.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

    # Condition on noise level, for each model input
    noise_level = torch.cat([noise_level] * num_prompts * 2)

    # Denoising Loop
    for i, t in enumerate(timesteps):
        # Logic to keep a component fixed to reference image
        if fixed_im is not None:
            # Inject noise
            alpha_cumprod = model.scheduler.alphas_cumprod[t]
            im_noisy = torch.sqrt(alpha_cumprod) * fixed_im + torch.sqrt(1 - alpha_cumprod) * torch.randn_like(fixed_im)

            # Replace component in noisy images with componen from fixed image
            # im_noisy_component = torch.zeros(im_noisy.shape, device=noisy_images.device, dtype=noisy_images.dtype)
            # noisy_images_component = torch.zeros(im_noisy.shape, device=noisy_images.device, dtype=noisy_images.dtype)
            # for i, view in enumerate(views):
            #     if i % 2 == 0:
            #         im_noisy_component += view.imprint(im_noisy)
            #     else:
            #         noisy_images_component += view.imprint(noisy_images[0])
            # noisy_images = im_noisy_component + noisy_images_component

            # # Correct for factor of 2 from view TODO: Fix this....
            # noisy_images = noisy_images[None] / len(views)
            im_noisy_component = views[0].inverse_view(im_noisy).to(noisy_images.device).to(noisy_images.dtype)
            noisy_images_component = views[1].inverse_view(noisy_images[0])
            noisy_images = im_noisy_component + noisy_images_component

            # Correct for factor of 2 from view TODO: Fix this....
            noisy_images = noisy_images[None] / 2.

        # Cat noisy image with upscaled conditioning image
        model_input = torch.cat([noisy_images, upscaled], dim=1)

        # Apply views to noisy_image
        viewed_inputs = []
        for view_fn in views:
            viewed_inputs.append(view_fn.view(model_input[0]))
        viewed_inputs = torch.stack(viewed_inputs)

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([viewed_inputs] * 2)
        model_input = model.scheduler.scale_model_input(model_input, t)

        # predict the noise residual
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=noise_level,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Invert the unconditional (negative) estimates
        # TODO: pretty sure you can combine these into one loop
        inverted_preds = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_uncond = torch.stack(inverted_preds)

        # Invert the conditional estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_text, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_text = torch.stack(inverted_preds)

        # Split predicted noise and predicted variances
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Combine noise estimates (and variance estimates)
        noise_pred = noise_pred.view(-1,num_prompts,3,height,width)
        predicted_variance = predicted_variance.view(-1,num_prompts,3,height,width)
        if reduction == 'mean':
            noise_pred = noise_pred.mean(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'alternate':
            noise_pred = noise_pred[:,i%num_prompts]
            predicted_variance = predicted_variance[:,i%num_prompts]

        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # compute the previous noisy sample x_t -> x_t-1
        noisy_images = model.scheduler.step(
            noise_pred, t, noisy_images, generator=generator, return_dict=False
        )[0]

    # Return denoised images
    return noisy_images