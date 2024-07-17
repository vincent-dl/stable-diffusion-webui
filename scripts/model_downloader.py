from datetime import datetime
import os
import requests


def model_downloader(model_url, saved_local_folder, saved_file_name=""):
    # Ensure the local folder exists
    os.makedirs(saved_local_folder, exist_ok=True)

    # Determine the filename from the URL
    filename = model_url.split("/")[-1]
    # Full path to save the file
    if len(saved_file_name) > 1:
        save_path = os.path.join(saved_local_folder, saved_file_name)
    else:
        save_path = os.path.join(saved_local_folder, filename)

    if not os.path.exists(save_path):
        print(f"Downloading {model_url} as {saved_file_name}")
        try:
            # Make a request to download the file
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes

            # Save the file
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            print(f"Model downloaded successfully and saved to {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the model: {e}")
    else:
        print(f"Model {model_url} existed. Skip downloading")


remake_lora_list = [
    (
        "https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors",
        "models/Lora",
        "EasyNegative.safetensors",
    ),
    ("https://civitai.com/api/download/models/20068", "models/Lora/", "badhandv4.pt"),
    (
        "https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt_version2.pt",
        "models/Lora",
        "bad_prompt_version2.pt",
    ),
    (
        "https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt.pt",
        "models/Lora",
        "",
    ),
    (
        "https://huggingface.co/nick-x-hacker/bad-artist/resolve/main/bad-artist.pt",
        "models/Lora",
        "",
    ),
    (
        "https://huggingface.co/nick-x-hacker/bad-artist/resolve/main/bad-artist-anime.pt",
        "models/Lora",
        "",
    ),
    (
        "https://huggingface.co/p1atdev/badquality/resolve/main/badquality.pt",
        "models/Lora",
        "",
    ),
    (
        "https://civitai.com/models/5224/bad-artist-negative-embedding",
        "models/Lora",
        "",
    ),
    (
        "https://civitai.com/api/download/models/20170",
        "models/Lora/",
        "bad-picture-chill-75v.pt",
    ),
    ("https://civitai.com/api/download/models/125849", "models/Lora", "bad-hands-5.pt"),
]

controlnet_model_list = [
    (
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_lineart.pth",
        "models/ControlNet",
        "control_v11p_sd15_lineart.pth",
    ),
    (
        "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin",
        "models/ControlNet",
        "ip-adapter-faceid-plus_sd15.bin"
    )
]
age_sliding_lora_list = [
    (
        "https://civitai.com/api/download/models/143150",
        "models/Lora/",
        "age_slider_v20.safetensors",
    ),
    (
        "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15_lora.safetensors",
        "models/Lora",
        "ip-adapter-faceid-plus_sd15_lora.safetensors",
    ),
]

sd_model_list = [
    (
        "https://huggingface.co/vule20/stable-diffusion-collection/resolve/main/weights/revAnimated_v122EOL.safetensors",
        "models/Stable-diffusion",
        "revAnimated_v122EOL.safetensors",
    ),
    (
        "https://civitai.com/api/download/models/125411",
        "models/Stable-diffusion",
        "realisticVisionV60B1_v60B1VAE.safetensors",
    ),
]

RED = "\033[31m"
RESET = "\033[0m"

for remake_lora in remake_lora_list:
    try:
        model_downloader(remake_lora[0], remake_lora[1], remake_lora[2])
    except Exception as e:
        print(f"{RED} {e} {RESET}")

for controlnet_model in controlnet_model_list:
    try:
        model_downloader(controlnet_model[0], controlnet_model[1], controlnet_model[2])
    except Exception as e:
        print(f"{RED} {e} {RESET}")

for age_sliding_lora in age_sliding_lora_list:
    try:
        model_downloader(age_sliding_lora[0], controlnet_model[1], controlnet_model[2])
    except Exception as e:
        print(f"{RED} {e} {RESET}")

for sd_model in sd_model_list:
    try:
        model_downloader(sd_model[0], sd_model[1], sd_model[2])
    except Exception as e:
        print(f"{RED} {e} {RESET}")