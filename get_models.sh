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
