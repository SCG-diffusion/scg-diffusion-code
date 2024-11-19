import torch
# from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL
from diffusers import ConsistencyDecoderVAE, AutoencoderKL
from pathlib import Path
from pipeline_scg_512 import PixArtAlphaPipeline
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16, use_safetensors=True)

pipe.to(device)


attn_res=(32,32)
steps_to_save_attention_maps = list(range(20)) # Steps to save attention maps
max_iter_to_alter = 10 # Which steps to stop updating the latents
refinement_steps = 20 # Number of refinement steps
scale_factor = 20 # Scale factor for the optimization step
iterative_refinement_steps = [0, 10, 20] # Iterative refinement steps
do_smoothing = True # Apply smoothing to the attention maps
smoothing_sigma = 0.5 # Sigma for the smoothing kernel
smoothing_kernel_size = 3 # Kernel size for the smoothing kernel
temperature = 0.5 # Temperature for the contrastive loss
softmax_normalize = False # Normalize the attention maps
softmax_normalize_attention_maps = False # Normalize the attention maps
add_previous_attention_maps = True # Add previous attention maps to the loss calculation
previous_attention_map_anchor_step = None # Use a specific step as the previous attention map
loss_fn = "ntxent" # Loss function to use
loss_mode = 'tv'


text_inputs = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=120,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
)


def get_token_indices_for_all_words(prompt, max_length=120):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    tokens = pipe.tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])
    tokens = [token for token in tokens if token != pipe.tokenizer.pad_token]

    words = prompt.split()
    word_token_indices = {}

    current_char_idx = 0
    token_index = 0
    for word_index, word in enumerate(words):
        word_start = prompt.find(word, current_char_idx)
        word_end = word_start + len(word)
        token_indices = []

        while token_index < len(tokens) and current_char_idx < word_end:
            clean_token = tokens[token_index].replace('▁', '') if '▁' in tokens[token_index] else tokens[token_index]
            token_length = len(clean_token)
            if word_start <= current_char_idx < word_end:
                token_indices.append(token_index)
            current_char_idx += token_length
            token_index += 1
        current_char_idx += 1
        word_token_indices[word_index] = token_indices

    return word_token_indices


prompts=["a blue apple and a pink bow"]
for prompt in prompts:
    # ratio=(0.35*item["0"],0.35*item["1"])
    ratio = (0,0)
    token_indices_dict = get_token_indices_for_all_words(prompt)
    token_indices = (token_indices_dict[2],token_indices_dict[6])
    color_indices = (token_indices_dict[1],token_indices_dict[5])
    # print(prompt)
    # print(token_indices)
    for seed in range(1):
        image = pipe(
                    prompt=prompt,
                    token_indices=token_indices,
                    color_indices=color_indices,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    num_inference_steps=50,
                    height = 512, 
                    width = 512,
                    attn_res=attn_res,
                    max_iter_to_alter = max_iter_to_alter,
                    refinement_steps=refinement_steps,
                    iterative_refinement_steps=iterative_refinement_steps,
                    scale_factor=scale_factor,
                    steps_to_save_attention_maps=steps_to_save_attention_maps,
                    do_smoothing=do_smoothing,
                    smoothing_kernel_size=smoothing_kernel_size,
                    smoothing_sigma=smoothing_sigma,
                    temperature=temperature,
                    softmax_normalize=softmax_normalize,
                    softmax_normalize_attention_maps=softmax_normalize_attention_maps,
                    add_previous_attention_maps=add_previous_attention_maps,
                    previous_attention_map_anchor_step=previous_attention_map_anchor_step,
                    loss_fn=loss_fn,
                    dab=True,
                    loss_mode=loss_mode,
                    is_cluster=True,
                    ratio=ratio
                    ).images[0]
        base_path = Path("./outputs")
        prompt_path = Path(prompt)
        full_path = base_path / prompt_path
        full_path.mkdir(parents=True, exist_ok=True)
        image.save(full_path / f'{seed}.png')
    