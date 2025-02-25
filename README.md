# Mixed-Diffusion

Anagrams + Sound in a single image!
Bad performance due to **latent** diffusion (VAE lack robustness).

## Main paramters to tune in /configs/main_denoise/main.yaml.trainer

<code> views </code> This is the anagram type you want to create. See /visual_anagrams/\_\_init\_\_.py for type list.

<code> anagram_balance_weight </code>

<code> image_prompt </code>

<code> audio_prompt </code>

<code> audio_weight </code>

## how to run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Kiaelen/Mixed-Diffusion/blob/main/exmaple_notebook.ipynb)
