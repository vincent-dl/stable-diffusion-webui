#!/bin/bash

echo "Download age sliding Lora"

AGE_SLIDER_LORA_FILE="models/Lora/age_slider_v20.safetensors"

# Check if the model file exists; if not, download it
if [ ! -f "${AGE_SLIDER_LORA_FILE}" ]; then
    echo "Downloading lora for age sliding..."
    wget https://civitai.com/api/download/models/143150 -O models/Lora/age_slider_v20.safetensors
else
    echo "Model already exists."
fi

REALISTIC_MODEL_FILE="models/Stable-diffusion/realisticVisionV60B1_v20Novae.safetensors"

echo "Downloading Realistic Vision V6.0 B1"

# Check if the model file exists; if not, download it
if [ ! -f "${REALISTIC_MODEL_FILE}" ]; then
    echo "Downloading lora for age sliding..."
    wget https://civitai.com/api/download/models/125411 -O models/Stable-diffusion/realisticVisionV60B1_v20Novae.safetensors
else
    echo "Model already exists."
fi

## ReV Animated
echo "Downloading ReV Animated V1.2.2-EOL stable diffusion"
wget https://civitai.com/api/download/models/46846 -O models/Stable-diffusion/revAnimated_v122EOL.safetensors

echo "Downloading EasyNegative LoRa for ReVAnimatedV1.2.2-EOL"
wget -P models/Lora https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors

echo "Downloading badhandv4 LoRa for ReVAnimatedV1.2.2-EOL"
wget https://civitai.com/api/download/models/20068 -O models/Lora/badhandv4.pt

echo "Downloading more LoRa for ReVAnimatedV1.2.2-EOL"
wget -P models/Lora https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors 
wget -P models/Lora https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt_version2.pt
wget -P models/Lora https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt.pt
wget -P models/Lora https://huggingface.co/nick-x-hacker/bad-artist/resolve/main/bad-artist.pt
wget -P models/Lora https://huggingface.co/nick-x-hacker/bad-artist/resolve/main/bad-artist-anime.pt
wget -P models/Lora https://huggingface.co/p1atdev/badquality/resolve/main/badquality.pt

echo "Downloading bad-picture negative embedding for ChilloutMix"
wget https://civitai.com/api/download/models/20170 -O models/Lora/bad-picture-chill-75v.pt
wget https://civitai.com/api/download/models/125849 -O models/Lora/bad-hands-5.pt