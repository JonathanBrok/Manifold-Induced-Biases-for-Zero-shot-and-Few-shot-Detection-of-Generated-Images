import os
import torch
from diffusers import KandinskyPipeline, KandinskyPriorPipeline, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPModel, AutoImageProcessor
from transformers import pipeline as pipeline_caption
from PIL import Image
import numpy as np
import torchvision
import imageio
from torchvision.transforms.functional import to_tensor
import json
import torchvision.transforms.functional as F


def my_flat(x):
    return x.view(x.size(0), -1).detach().cpu()
        
def get_sd_v14_model(device):
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    pipeline.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    unet = pipeline.unet.to(device).eval().half()
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device).half()
    vae = pipeline.vae.to(device).eval().half()
    return unet, tokenizer, text_encoder, vae, noise_scheduler

def numpy_chunk(arr, num_chunks, axis=0):
    """
    Splits a NumPy array into approximately equal chunks along the specified axis.
    
    Parameters:
    - arr: NumPy array to be split
    - num_chunks: Number of chunks to create
    - axis: Axis along which to split the array
    
    Returns:
    - List of NumPy arrays
    """
    # Calculate the size of each chunk
    chunk_size = arr.shape[axis] // num_chunks
    remainder = arr.shape[axis] % num_chunks
    chunks = []
    
    # Determine indices to split the array
    indices = np.cumsum([0] + [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)])
    
    # Use np.split to split the array at the calculated indices
    splits = np.split(arr, indices[1:-1], axis=axis)
    
    return splits

def my_resize(img_t, siz):
    """
    Resize and crop an image tensor to the specified size.
    
    Parameters:
    - img_t (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
    - siz (int): Target size for the cropped square (output dimensions will be siz x siz).
    
    Returns:
    - torch.Tensor: Resized and cropped tensor of shape (C, siz, siz) or (B, C, siz, siz).
    """
    
    img_t = torchvision.transforms.Resize(siz+3)(img_t)
    
    # Manually calculate center crop to siz x siz
    start_x = (img_t.size(-1) - siz) // 2
    start_y = (img_t.size(-2) - siz) // 2

    if img_t.dim() == 3:  # CHW format
        img_t = img_t[:, start_y:start_y + siz, start_x:start_x + siz]
    elif img_t.dim() == 4:  # BCHW format
        img_t = img_t[:, :, start_y:start_y + siz, start_x:start_x + siz]
    else:
        raise ValueError("Unsupported tensor shape: {}".format(img_t.shape))

    return img_t

def load_and_convert_image_batch(image_paths, device):
    """
    Loads a batch of images from file paths and returns them as a list of tensors.

    Parameters:
    - image_paths (list of str): List of image file paths.
    - device (torch.device): Device to which the tensors should be moved (e.g., 'cpu' or 'cuda').

    Returns:
    - list of torch.Tensor: List of tensors, each of shape (H, W, C), dtype `uint8`, with values in the range [0, 255].
    """
    return [torch.tensor(imageio.imread(path, pilmode='RGB')) for path in image_paths]

def preprocess_image_batch(batch_img_t, siz, device):
    """
    Preprocesses a batch of image tensors for input to a neural network.

    Parameters:
    - batch_img_t (list of torch.Tensor): List of tensors, each of shape (H, W, C), with values in the range [0, 255].
    - siz (int, optional): Target size for resizing and cropping. Default is 512.

    Returns:
    - torch.Tensor: Preprocessed tensor of shape (B, 3, siz, siz), dtype `float16`, with values scaled to the range [-1, 1].
    """
    batch_img_t = [img.permute(2, 0, 1).to(device).half() for img in batch_img_t]
    batch_img_t = [my_resize(img, siz) for img in batch_img_t]
    batch_img_t = torch.stack(batch_img_t)
    # Scale dynamic range [0, 255] -> [-1, 1]
    batch_img_t = 2 * (batch_img_t / 255.0) - 1
    return batch_img_t

def postprocess_image(img_t, siz, do_resize=True):
    if do_resize:
        img_t = my_resize(img_t, siz)
    img_t = (img_t / 2 + 0.5).clamp(0, 1) * 255
    img_t = img_t.detach().cpu()
    img_t = img_t.permute(0, 2, 3, 1).float().numpy()
    img_t = img_t.round().astype("uint8")  # [0]
    return img_t

def normalize_batch(batch, epsilon=1e-8):
    """normalize each element in a pytorch batch (dim 0 is the batch dimension)"""
    # Normalize this tensor without assuming its element dimensionality
    dims_to_normalize = tuple(range(1, batch.dim()))  # Create a tuple of dimensions excluding the batch dimension
    # normalize each batch of noise by its norm
    norms = torch.norm(batch, p=2, dim=dims_to_normalize, keepdim=True)  # calculate the L2 norm per-element-in-batch 
    return  batch / (norms + epsilon)

def factory_sdv14_based_criterion(device, num_noise, epsilon_reg, time_frac, tokenizer, text_encoder, image_to_text, vae, scheduler, unet, clip, processor, cos, siz, prompts_list=None):

    def sdv14_based_criterion(images_raw):
        num_images = len(images_raw)
        
        images = preprocess_image_batch(images_raw, siz, device)
        
        # # Load images, pre-process and and batch
        # images = torch.concatenate([preprocess_image(path, device, siz) for path in image_paths])
        
        if prompts_list is not None:  # manual caption
            if len(prompts_list) == 1:  # same caption for all images
                prompts = prompts_list * num_images
            else:
                assert len(prompts_list) == num_images  # caption per-image TODO test this, it was not tested yet
                prompts = prompts_list
        else:  # Auto-caption
            prompts = [
                image_to_text(F.to_pil_image(cur_image_raw.permute(2, 0, 1)), prompt="<image>\nUSER: Generate a caption for the image that contains only facts and detailed\nASSISTANT:", generate_kwargs={"max_new_tokens": 76})[0]
                .get("generated_text").split("ASSISTANT:")[1]
                for cur_image_raw in images_raw
            ]

        # Get text embeddings hidden state
        if True: # Option A: # Repeat the prompts
            expanded_prompts = []
            for prompt in prompts:
                expanded_prompts.extend([prompt] * num_noise)
            text_tokens = tokenizer(expanded_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            input_ids = text_tokens.input_ids.to(device)
            text_emb = text_encoder(input_ids).last_hidden_state
        else:  # Option B: # Repeat the embeddings
            
            text_tokens = tokenizer(prompts, padding="max_length", max_length=77, return_tensors="pt")
            input_ids = text_tokens.input_ids.to(device)
            text_emb = text_encoder(input_ids).last_hidden_state
            text_emb = text_emb.repeat_interleave(num_noise, dim=0)
        
        # Step 2: Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()  # it's actually latents by the end of this line..
            latents = latents * vae.config.scaling_factor
            
        latents = latents.repeat_interleave(num_noise, dim=0).half()  # [num_images*num_noise, 4, 64, 64 ]
        
        # Step 3: Draw precise-approximation-that-is-Gaussian-interchangeable spherical random tensor
        gauss_noise = torch.randn_like(latents, device=device).half()  # draw normal noise
        spherical_noise = normalize_batch(gauss_noise, epsilon_reg).half()  # apply normalization for 1-sphere uniform
        sqrt_d = torch.prod(torch.tensor(latents.shape[1:])).float().sqrt()  # scalar factor to keep scale consistent after applying normalization, obtained as the the known expectation of the norm
        spherical_noise *= sqrt_d  # scale back
        timestep = time_frac * scheduler.config.num_train_timesteps  # 100.0 for time_frac==0.1 (noise_scheduler.config.num_train_timesteps == 0)
        timestep = torch.full((latents.shape[0],), timestep, device=device, dtype=torch.long)
        
        # add noise
        noisy_latents = scheduler.add_noise(original_samples=latents, noise=spherical_noise, timesteps=timestep).half()
        
        # print alpha_t and sqrt(d). For input image resized to 512 we have a 4 X 64 X 64 latent space, flattened to d = 16384
        alpha_t = scheduler.alphas_cumprod[timestep[0].item()]  # Extract alpha_t from the scheduler
        print(f"timestep: {timestep[0].item()}, alpha_t: {alpha_t.item()}")
        print(f"D (dimension of latent space): {torch.prod(torch.tensor(latents.shape[1:])).item()}, sqrt(D): {sqrt_d.item()}")

        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        noise_pred = 1 / vae.config.scaling_factor * noise_pred
        
        
        ### Steps 5-7: Compare predictions of noise with image - map both to CLIP (decode from latent to image space then embed to CLIP)
        
        # try to clear memmory
        del noisy_latents, timestep, gauss_noise, text_emb, input_ids, images
        # torch.cuda.empty_cache()

        sub_batch_size = 4  # Maximal batch size in vae.decoder on a single A100 GPU
        if noise_pred.size(0) <= sub_batch_size:  # inference regularly
            decoded_noise = vae.decode(noise_pred, return_dict=False)[0]
            decoded_spherical_noise = vae.decode(spherical_noise, return_dict=False)[0]
        
        else:  # split batch to avoid errors
            
            num_sub_batches = (noise_pred.size(0) + sub_batch_size - 1) // sub_batch_size  # Calculate the number of sub-batches
            print(f'batch of {noise_pred.size(0)}, is split to {num_sub_batches} sub-batches of (max) {sub_batch_size}')
            
            decoded_noise_list = []
            decoded_spherical_noise_list = []
            with torch.no_grad():
                for i in range(num_sub_batches):
                    print(f'{i}-th sub-batch')
                    start_idx = i * sub_batch_size
                    end_idx = min((i + 1) * sub_batch_size, noise_pred.size(0))
                    # Ensure GPU memory is cleared before decoding
                    torch.cuda.empty_cache()
                    decoded_noise_sub_batch = vae.decode(noise_pred[start_idx:end_idx], return_dict=False)[0]
                    decoded_sphrical_noise_sub_batch = vae.decode(spherical_noise[start_idx:end_idx], return_dict=False)[0]
                    
                    decoded_noise_list.append(decoded_noise_sub_batch)
                    decoded_spherical_noise_list.append(decoded_sphrical_noise_sub_batch)
                    
                    del decoded_noise_sub_batch, decoded_sphrical_noise_sub_batch
                    
            decoded_noise = torch.cat(decoded_noise_list, dim=0)  # B, 3, siz, siz torch
            decoded_spherical_noise = torch.cat(decoded_spherical_noise_list, dim=0) 

        
        
        decoded_noise = postprocess_image(decoded_noise, siz)  # B, siz, siz, 3 numpy
        decoded_spherical_noise = postprocess_image(decoded_spherical_noise, siz)
        
        # decoded_noise_chunks = torch.chunk(decoded_noise, chunks=num_images, dim=0)    
        decoded_noise_chunks = numpy_chunk(decoded_noise, num_chunks=num_images)
        decoded_spherical_noise_chunks = numpy_chunk(decoded_spherical_noise, num_chunks=num_images)
        
        ret_dicts = []
        for cur_decoded_noise_chunk, cur_dec_spherical_chunk, cur_image_raw in zip(decoded_noise_chunks, decoded_spherical_noise_chunks, images_raw):
            
            # similarity in CLIP space 
            img_s = cur_image_raw.float().to(device)
            
            img_in = processor(images=img_s, return_tensors="pt").to(device)
            img_clip = clip.get_image_features(**img_in).detach().cpu()
            
            img_d_in = processor(images=cur_decoded_noise_chunk, return_tensors="pt").to(device)
            img_d_clip = clip.get_image_features(**img_d_in).detach().cpu()
            
            img_s_in = processor(images=cur_dec_spherical_chunk, return_tensors="pt").to(device)
            img_s_clip = clip.get_image_features(**img_s_in).detach().cpu()
            
            bias_vec = cos(img_clip, img_d_clip).numpy()
            kappa_vec = cos(img_d_clip, img_s_clip).numpy()        
            D_vec = torch.norm(img_d_clip.view(img_d_clip.size(0), -1), p=2, dim=1).cpu().numpy()

            bias_mean = bias_vec.mean()
            kappa_mean = kappa_vec.mean()
            D_mean = D_vec.mean()
            
            # criterion using coefficients sqrt(d), 1, -1 as in paper
            d_clip = 512
            sqrt_d_clip = d_clip ** 0.5
            criterion = 1 + (sqrt_d_clip * bias_mean - D_mean + kappa_mean) / (sqrt_d_clip + 2)
            
            cur_dict = {
                "criterion": float(criterion),
                "bias": float(bias_mean),
                "kappa": float(kappa_mean),
                "D": float(D_mean),
            }
            
            ret_dicts.append(cur_dict)

        return ret_dicts
        
    return sdv14_based_criterion
   
def load_sdv14_criterion_functionalities(device):
    # load main functionalities
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    unet, tokenizer, text_encoder, vae, scheduler = get_sd_v14_model(device)
    image_to_text = pipeline_caption("image-to-text", model="llava-hf/llava-1.5-7b-hf", device=device)
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    
    
    return unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text
      
if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_path_1 = "example_images/real_car.jpg"  # "example_images/gen_frog.png"  # "example_images/real_car.jpg"
    image_path_2 = "example_images/gen_okapi.png"  # "example_images/real_dog.png"   # "example_images/gen_okapi.png"
    siz = 512
    image_type = 0
    dataset_type = 'sanity'  # 'train' or 'test' usualy
    dataset_name = 'my_collection'  # generative technique or source of real data usually
    num_noise = 2  # maximal for single A100 and 2 images (totaling in  batch of 2*256=512). We have sub-batches for the vae.decoder, so the bottleneck is the unet
    time_frac = 0.01
    epsilon_reg = 1e-8
    
    # load functionalities and models
    unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text = load_sdv14_criterion_functionalities(device)
    
    # Example run
    image_paths = [image_path_1, image_path_2]
    sdv14_based_criterion = factory_sdv14_based_criterion(device, num_noise, epsilon_reg, time_frac, tokenizer, text_encoder, image_to_text, vae, scheduler, unet, clip, processor, cos, siz)
    
    # load images and convert to tensor (still [0,255] range etc..)
    images = load_and_convert_image_batch(image_paths, device)
    
    # run the criterion on the image batch
    res_dict_list = sdv14_based_criterion(images)
    
    # print results
    for idx, (cur_dict, cur_path) in enumerate(zip(res_dict_list, image_paths)):
        print('Image:')
        print(cur_path)
        for key, value in cur_dict.items():
            print(key)
            print(value)
                
    
        
        