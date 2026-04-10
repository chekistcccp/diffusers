import os
import re
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import safetensors
from safetensors.torch import load_file
import random
import torch
import yaml
from collections import OrderedDict
from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from pretrained_encoder.inference import speciesModel
from pretrained_encoder.tokenizer import ProtTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--embeddings_path', type=str, required=True)
parser.add_argument('--local_model_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--ckpt', type=int, required=True)
parser.add_argument('--fname_prefix', type=str, default='bird')
parser.add_argument('--species_encoder_checkpoint', type=str, required=True)
args = parser.parse_args()

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

def resolve_config_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((config_base_dir / path).resolve())

config['embeddings'] = resolve_config_path(args.embeddings_path)
config['species_classes'] = resolve_config_path(config['species_classes'])
config['vocabulary'] = resolve_config_path(config['vocabulary'])

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

prot_encoder = speciesModel(
    config['max_tokens'],
    config['embeddings'],
    config['species_classes'],
    config['vocabulary'],
    checkpoint_path=resolve_config_path(args.species_encoder_checkpoint),
    device=device,
)
prot_encoder.to(device)
tokenizer = prot_encoder.tokenizer

pipe = DiffusionPipeline.from_pretrained(
    args.local_model_path, low_cpu_mem_usage=False, safety_checker=None
)

model_path = f"{args.model_path}/checkpoint-{args.ckpt}"
pipe.text_encoder = prot_encoder
pipe.tokenizer = tokenizer
pipe.load_lora_weights(model_path)
pipe.fuse_lora()
pipe.to(device)

outf = f"{args.output_dir}/ckpt-{args.ckpt}"
os.makedirs(outf, exist_ok=True)

for idx in range(len(df_test)):
    try:
        prompt_seq = df_test.loc[idx, 'prompt_seq']
        spe_name = df_test.loc[idx, 'species']
        image = pipe(prompt_seq, num_inference_steps=50, guidance_scale=9.5).images[0]
        fname = f"{args.fname_prefix}_test_{str(idx)}_{spe_name}_ckpt-{args.ckpt}.jpeg"
        image.save(os.path.join(outf, fname))
        print(f"run prediction for {spe_name}-{idx}")
    except Exception as e:
        print(f"error for {spe_name}-{idx}: {e}")
