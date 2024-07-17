import os
from workers import api


webui_server_url = "http://69.197.187.29:7860"
out_dir = "outputs"
sd_api_client = api.StableDiffusionAPIClient(
    server_url=webui_server_url, output_dir=out_dir
)

age_sliding_sample_image = "extensions/sample_images/Jensen-NVIDIA.png"
sd_api_client.age_slider(
    image_file=age_sliding_sample_image,
    prompt="asian man, white shirt, grey jacket, black hair, portrait, looking at viewer, forest, hat<lora:age_slider_v20:-1>",
    negative_prompt="nude, breasts, topless, cartoon, cgi, render, illustration, painting, drawing, bad quality, grainy, low resolution",
    batch_size=4,
)


remake_sample_image = "extensions/sample_images/sample_woman.png"
sd_api_client.remake_image(image_file=remake_sample_image, batch_size=4)
