import os
import argparse
import pandas as pd
import torch
import yaml
from pathlib import Path
from diffusers import DiffusionPipeline
from pretrained_encoder.protein_clip_adapter import ProteinCLIPAdapter, get_device, empty_cache

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--embeddings_path', type=str, required=True)
parser.add_argument('--local_model_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--ckpt', type=int, required=True)
parser.add_argument('--fname_prefix', type=str, default='bird')
parser.add_argument('--adapter_checkpoint', type=str, default=None, help='Path to protein_clip_adapter.pt')
parser.add_argument('--guidance_scale', type=float, default=7.5)
args = parser.parse_args()


def resolve_config_path(base_dir, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


df = pd.read_csv(args.data_file, sep='\t', header=None)
df.columns = ['file_path', 'prompt_seq']

def extract_species(file_path):
    parent_name = Path(file_path).parent.name
    if parent_name:
        return parent_name
    return "unknown"

df['species'] = df['file_path'].apply(extract_species)

df_test = df.groupby('species').head(20)
df_test = df_test.reset_index(drop=True)

with open(args.config_file, 'rb') as f:
    config = yaml.safe_load(f)

config_base_dir = Path(args.config_file).resolve().parent
config['embeddings'] = resolve_config_path(config_base_dir, args.embeddings_path)
config['species_classes'] = resolve_config_path(config_base_dir, config['species_classes'])
config['vocabulary'] = resolve_config_path(config_base_dir, config['vocabulary'])
local_model_path = resolve_config_path(config_base_dir, args.local_model_path)
model_path = resolve_config_path(config_base_dir, args.model_path)
output_dir = resolve_config_path(config_base_dir, args.output_dir)

device = get_device()

text_encoder = ProteinCLIPAdapter(
    pretrained_model_name_or_path=local_model_path,
    embeddings_dir=config['embeddings'],
    vocabulary_path=config['vocabulary'],
    species_classes_path=config['species_classes'],
    max_tokens=config['max_tokens'],
    device=device,
)

if args.adapter_checkpoint is not None:
    adapter_ckpt_path = args.adapter_checkpoint
    if not os.path.isabs(adapter_ckpt_path):
        adapter_ckpt_path = os.path.abspath(adapter_ckpt_path)
    if os.path.isfile(adapter_ckpt_path):
        adapter_state_dict = torch.load(adapter_ckpt_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_ckpt_path}")
    text_encoder.load_state_dict(adapter_state_dict, strict=False)
    print(f"Loaded adapter weights from {adapter_ckpt_path}")

text_encoder.to(device)
text_encoder.eval()
tokenizer = text_encoder.tokenizer

pipe = DiffusionPipeline.from_pretrained(
    local_model_path, low_cpu_mem_usage=False, safety_checker=None
)

pipe.text_encoder = text_encoder
pipe.tokenizer = tokenizer

checkpoint_path = f"{model_path}/checkpoint-{args.ckpt}"
pipe.load_lora_weights(checkpoint_path)
pipe.fuse_lora()
pipe.to(device)
pipe.set_progress_bar_config(disable=True)


def protein_encode_prompt(self_pipe, prompt, device, num_images_per_prompt, do_classifier_free_guidance=False, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, lora_scale=None, clip_skip=None):
    if prompt_embeds is not None:
        return prompt_embeds, negative_prompt_embeds

    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)

    tokenized = text_encoder.tokenizer(
        prompt,
        max_length=text_encoder.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    text_embeddings = text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)[0]
    prompt_embeds = text_embeddings.to(dtype=text_encoder.dtype)

    # Duplicate prompt_embeds for each image per prompt
    if num_images_per_prompt > 1:
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    if do_classifier_free_guidance:
        # Build the unconditional embedding through the SAME adapter path as the
        # conditional one (empty protein -> all-PAD input -> compression -> SOS/EOS
        # -> position embed -> CLIP encoder -> final_layer_norm). The previous
        # implementation encoded "" through the raw CLIP tokenizer + CLIP text model,
        # which places uncond in a different distribution than cond, so the CFG
        # direction (cond - uncond) points off-manifold and fails to steer global
        # structure. Sharing the adapter path keeps both branches comparable.
        uncond_tokens = [""] * batch_size
        uncond_tokenized = text_encoder.tokenizer(
            uncond_tokens,
            max_length=text_encoder.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_tokenized["input_ids"].to(device)
        uncond_attention_mask = uncond_tokenized["attention_mask"].to(device)
        uncond_embeddings = text_encoder(
            uncond_input_ids, attention_mask=uncond_attention_mask, return_dict=True
        )[0].to(dtype=text_encoder.dtype)

        if num_images_per_prompt > 1:
            uncond_embeddings = uncond_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        prompt_embeds = torch.cat([uncond_embeddings, prompt_embeds])

    return prompt_embeds, negative_prompt_embeds

import types
pipe.encode_prompt = types.MethodType(protein_encode_prompt, pipe)

outf = f"{output_dir}/ckpt-{args.ckpt}"
os.makedirs(outf, exist_ok=True)

for idx in range(len(df_test)):
    try:
        prompt_seq = df_test.loc[idx, 'prompt_seq']
        spe_name = df_test.loc[idx, 'species']
        image = pipe(prompt_seq, num_inference_steps=50, guidance_scale=args.guidance_scale).images[0]
        fname = f"{args.fname_prefix}_test_{str(idx)}_{spe_name}_ckpt-{args.ckpt}.jpeg"
        image.save(os.path.join(outf, fname))
        print(f"run prediction for {spe_name}-{idx}")
    except Exception as e:
        print(f"error for {spe_name}-{idx}: {e}")
