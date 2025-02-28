# Manifold Induced Biases for Zero-shot and Few-shot detection of Generated Images

## Overview
This is the official implementation of the Manifold Bias Criteria, presented in Brokman et. al, ICLR 2025. These are zero-shot criteria for detecting AI-generated images.

## Features
- **Zero-shot detection**: No need for labeled generated images.
- **Theoretically grounded**:  Based on the mathematical analysis of biases within pre-trained diffusion models
- **Out-performs existing zero-shot methods** in generalization across multiple generative techniques.

## Requirements
This implementation is built using PyTorch and Hugging Face's `diffusers` and `transformers`. Ensure you have the required dependencies installed before running the code.

### Install Dependencies
```bash
pip install torch torchvision diffusers transformers numpy imageio
```

## Usage

### 1. Load the Model and Utilities
```python
from manifold_bias_criteria import load_sdv14_criterion_functionalities, factory_sdv14_based_criterion, load_and_convert_image_batch
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text = load_sdv14_criterion_functionalities(device)
```

### 2. Define Parameters
```python
num_noise = 2  # Number of spherical noise perturbations
time_frac = 0.01  # Fraction of diffusion time for perturbations
epsilon_reg = 1e-8  # Regularization parameter
siz = 512  # Image resizing dimension

sdv14_based_criterion = factory_sdv14_based_criterion(
    device, num_noise, epsilon_reg, time_frac, tokenizer, text_encoder,
    image_to_text, vae, scheduler, unet, clip, processor, cos, siz
)
```

### 3. Run Detection on Images
```python
image_paths = ["example_images/fake.png", "example_images/real.png"]
images = load_and_convert_image_batch(image_paths, device)
res_dict_list = sdv14_based_criterion(images)

for idx, (cur_dict, cur_path) in enumerate(zip(res_dict_list, image_paths)):
    print(f'Image: {cur_path}')
    for key, value in cur_dict.items():
        print(f'{key}: {value}')
```

## Output
The script returns a dictionary with the following keys for each image:
- **criterion**: The computed detection score.
- **bias**: Measures the alignment between bias in noise prediction, and .
- **kappa**: Quantifies the curvature in the probability manifold.
- **D**: Gradient magnitude of the score function.

A lower `criterion` score suggests that the image is more likely to be real, while a higher score indicates a generated image.

## Example Run
```bash
python manifold_bias_criteria.py
```
