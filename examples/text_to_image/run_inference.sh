#!/bin/bash

# Run the inference script

python inference.py \
    --data_file /media/ssd1/guojirui/diffuser2/data/20260517/large_species_img_w_prot_list_16birds_subsystems_resize_512.validate \
    --config_file pretrained_encoder/config.yml \
    --embeddings_path /media/ssd1/guojirui/diffuser2/data/20260517/16embeddings \
    --local_model_path /media/ssd1/guojirui/diffuser2/data/stable-diffusion-v1-4 \
    --model_path /media/ssd1/guojirui/diffuser/diffusers-bird-lora/examples/text_to_image/20260525-16-rank4 \
    --ckpt 1000000 \
    --adapter_checkpoint alignment_output/best_model/protein_clip_adapter.pt \
    --output_dir inference_results \
    --guidance_scale 7.5


#python inference2.py \
    #--data_file "/media/hit/1eb711ef-98e1-4d1d-bf78-d6f768e27258/guo/dataprocess/20260318/bird10/large_species_img_w_prot_list_10birds_subsystems_resize_512.validate" \
    #--config_file "./config_10birds.yml" \
    #--embeddings_path "/media/hit/1eb711ef-98e1-4d1d-bf78-d6f768e27258/guo/dataprocess/20260318/10bird/10embeddings" \
    #--local_model_path "/media/hit/1eb711ef-98e1-4d1d-bf78-d6f768e27258/guo/data/stable-diffusion-v1-4" \
    #--model_path "./20260324-10-rank4/" \
    #--output_dir "./validate/20260324-10-rank4" \
    #--ckpt 1000000 \
    #--fname_prefix "bird-10"