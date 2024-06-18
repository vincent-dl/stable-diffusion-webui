from datetime import datetime
import urllib.request
import base64
import json
import time
import os


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(webui_server_url, api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(webui_server_url, output_dir, **payload):
    response = call_api(webui_server_url, 'sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(output_dir, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def call_img2img_api(webui_server_url, output_dir, **payload):
    response = call_api(webui_server_url, 'sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(output_dir, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)
        
        
def payloader(
    image_file,
    prompt,
    negative_prompt,
    batch_size=1,
    steps=20,
    n_iter=1,
    seed=-1,
    sd_model_checkpoint="v1-5-pruned.safetensors",
    sampler_name="DPM++ 2M SDE",
    width=512,
    height=512,
    cfg_scale=7,
    ):
    """
    This function takes in parameters and return prompt for the server.
    """
    reactor_args=[
        encode_file_to_base64(image_file), #0
        True, #1 Enable ReActor
        '0', #2 Comma separated face number(s) from swap-source image
        '0', #3 Comma separated face number(s) for target image (result)
        'inswapper_128.onnx', #4 model path
        'CodeFormer', #4 Restore Face: None; CodeFormer; GFPGAN
        1, #5 Restore visibility value
        True, #7 Restore face -> Upscale
        '4x_NMKD-Superscale-SP_178000_G', #8 Upscaler (type 'None' if doesn't need), see full list here: http://127.0.0.1:7860/sdapi/v1/script-info -> reactor -> sec.8
        1.5, #9 Upscaler scale value
        1, #10 Upscaler visibility (if scale = 1)
        False, #11 Swap in source image
        True, #12 Swap in generated image
        1, #13 Console Log Level (0 - min, 1 - med or 2 - max)
        0, #14 Gender Detection (Source) (0 - No, 1 - Female Only, 2 - Male Only)
        0, #15 Gender Detection (Target) (0 - No, 1 - Female Only, 2 - Male Only)
        False, #16 Save the original image(s) made before swapping
        0.8, #17 CodeFormer Weight (0 = maximum effect, 1 = minimum effect), 0.5 - by default
        False, #18 Source Image Hash Check, True - by default
        False, #19 Target Image Hash Check, False - by default
        "CPU", #20 CPU or CUDA (if you have it), CPU - by default
        True, #21 Face Mask Correction
        0, #22 Select Source, 0 - Image, 1 - Face Model, 2 - Source Folder
        None, #23 Filename of the face model (from "models/reactor/faces"), e.g. elena.safetensors, don't forger to set #22 to 1
        None, #24 The path to the folder containing source faces images, don't forger to set #22 to 2
        None, #25: Multiple Source Images skip it for API
        True, #26 Randomly select an image from the path
        True, #27 Force Upscale even if no face found
        0.6, #28 Face Detection Threshold
        2, #29 Maximum number of faces to detect (0 is unlimited)
    ]
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": n_iter,
        "batch_size": batch_size,
        # "init_images": init_images,
        # example args for Refiner and ControlNet
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "batch_images": "",
                        "control_mode": control_mode,
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "image": {
                            "image": encode_file_to_base64(image_file),
                            "mask": None,  # base64
                        },
                        "input_mode": "simple",
                        "is_ui": True,
                        "loopback": False,
                        "model": model,
                        "module": module,
                        "output_dir": "",
                        "pixel_perfect": False,
                        "processor_res": 512,
                        "resize_mode": resize_mode,
                        "threshold_a": 100,
                        "threshold_b": 200,
                        "weight": 1,
                    }
                ]
            },
        },
        "override_settings": {
            "sd_model_checkpoint": sd_model_checkpoint,
        },
    }
    return payload
