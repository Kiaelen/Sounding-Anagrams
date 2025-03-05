# Sounding Anagrams

**Anagram + Sound** in a single image!

Adapted from [image-that-sound](https://github.com/IFICL/images-that-sound) and [visual anagrams](https://github.com/dangeng/visual_anagrams).
<hr>

## Results

Download **audio** results in [assets](https://github.com/Kiaelen/Sounding-Anagrams/tree/main/assets).

### Gaussian Filter Example

  **image_prompt:**
  - a castle with bell towers, grayscale
  - an antique model car, grayscale

**audio_prompt:** bird chirping

<div align="center">
  <img width="100%" src="assets\gaussian\car-castle-bird\view0.img.png">
  <img width="30%" src="assets\gaussian\car-castle-bird\view1.img.png">
</div>

### Patch Permute Example

  **image_prompt:**
  - painting of trees, lithograph style, grayscale
  - painting of castles, lithograph style, grayscale

**audio_prompt:** bell ringing

<div align="center">
  <img width="100%" src="assets\permute\tree-castle-bell\view0.img.png">
  <img width="100%" src="assets\permute\tree-castle-bell\view1.img.png">
</div>

### Patch Rotate Example

  **image_prompt:**
  - painting of a garden, lithograph style, grayscale
  - paiting of a helicopter, lithograph style, grayscale ~~(the result is a plane)~~

**audio_prompt:** bird chirping

<div align="center">
  <img width="100%" src="assets\rotate\trees-helicopter-bird\view0.img.png">
  <img width="100%" src="assets\rotate\trees-helicopter-bird\view1.img.png">
</div>

## Main paramters to tune in [config](https://github.com/Kiaelen/Sounding-Anagrams/blob/main/configs/main_denoise/main.yaml).trainer

<code> views </code> This is the anagram type you want to create. See /visual_anagrams/\_\_init\_\_.py for type list.

<code> anagram_balance_weight </code>

<code> image_prompt </code>

<code> audio_prompt </code>

<code> audio_weight </code>
<hr>

## Limitations

Good results extremely hard to obtain. Have to painstakingly tune hyperparameters for half an hour on average to obtain a relatively good sounding anagram.

VAE Decoder cannot preserve boundaries properly as can be seen in the patch permute examples.

Black strip identifid in generated images as Auffusion denoising results.

## How to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cj_SOwWNtn4z0XPShqs0Qu-Ib4lMhRis?usp=sharing)
