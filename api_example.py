import os
from workers import api


webui_server_url = "http://69.197.148.156:7860"
out_dir = "outputs"
sd_api_client = api.StableDiffusionAPIClient(
    server_url=webui_server_url, output_dir=out_dir
)

age_sliding_sample_image = "extensions/sample_images/Jensen-NVIDIA.png"

sd_api_client.age_slider(
    image_file=age_sliding_sample_image,
    age=6,
    lora_range=0.8,
    gender="man",  # woman will be woman or girl
    prompt="asian man, white shirt, grey jacket, black hair, portrait, looking at viewer, forest",
    negative_prompt="nude, breasts, topless, cartoon, cgi, render, illustration, painting, drawing, bad quality, grainy, low resolution",
    batch_size=4,
)
