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
scale_factor = 20 # Scale factor for the optimization step


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

seed_list=[1]
prompts=["a pixar-style girl and a watercolor countryside"]
for prompt in prompts:
    # ratio=(0.35*item["0"],0.35*item["1"])
    ratio = (0,0)
    token_indices_dict = get_token_indices_for_all_words(prompt)
    token_indices = (token_indices_dict[2],token_indices_dict[6])
    color_indices = (token_indices_dict[1],token_indices_dict[5])
    # print(prompt)
    # print(token_indices)
    for seed in seed_list:
        image = pipe(
                    prompt=prompt,
                    token_indices=token_indices,
                    color_indices=color_indices,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    num_inference_steps=50,
                    height = 512, 
                    width = 512,
                    attn_res=attn_res,
                    scale_factor=scale_factor,
                    dab=True,
                    is_cluster=True,
                    ratio=ratio
                    ).images[0]
        base_path = Path("./outputs")
        prompt_path = Path(prompt)
        full_path = base_path / prompt_path
        full_path.mkdir(parents=True, exist_ok=True)
        image.save(full_path / f'{seed}.png')
    
