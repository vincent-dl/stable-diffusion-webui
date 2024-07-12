from datetime import datetime
import urllib.request
import base64
import json
import time
import os


class StableDiffusionAPIClient:
    def __init__(self, server_url="http://localhost:7860", output_dir="outputs"):
        self.server_url = server_url
        self.output_dir = output_dir
        self.seed = -1  # random seed
        self.steps = 30  # steps for running SD
        self.height = 512
        self.width = 512
        self.cfg_scale = 7
        self.sampler_name = "DPM++ 2M"
        self.scheduler = "karras"
        self.n_iter = 1
        self.batch_size = 1
        self.out_dir_t2i = os.path.join(self.output_dir, "txt2img")
        os.makedirs(self.out_dir_t2i, exist_ok=True)

        # Here we define a list of loras, and model for different applications

        # To get a list of available samples, run:
        # curl -X 'GET' \
        #     'http://localhost:7861/sdapi/v1/samplers' \
        #     -H 'accept: application/json'

    def _timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    def _encode_file_to_base64(self, path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def _decode_and_save_base64(self, base64_str, save_path):
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))

    def _call_api(self, api_endpoint, **payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}/{api_endpoint}",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))

    def _call_txt2img_api(self, **payload):
        response = self._call_api("sdapi/v1/txt2img", **payload)
        for index, image in enumerate(response.get("images")):
            save_path = os.path.join(
                self.out_dir_t2i, f"txt2img-{self._timestamp()}-{index}.png"
            )
            self._decode_and_save_base64(image, save_path)

    def age_slider(
        self,
        image_file="extensions/sd-webui-reactor/example/IamSFW.jpg",
        prompt="asian man, white shirt, grey jacket, black hair, portrait, looking at viewer, forest, hat<lora:age_slider_v20:-1>",
        negative_prompt="nude, breasts, topless, cartoon, cgi, render, illustration, painting, drawing, bad quality, grainy, low resolution",
        seed=-1,
        steps=30,
        width=512,
        height=512,
        denoising_strength=0.75,
        cfg_scale=7,
        sampler_name="DPM++ 2M",
        scheduler="karras",  # option exponential
        n_iter=1,
        batch_size=1,
        sd_model_checkpoint="v1-5-pruned.safetensors",
    ):
        # if you wanna know how do I get these arguments, so can visit
        # web_url:scripts-info to learn more about. For example, if your server
        # url is http://localhost:7860, then navigate to
        # http://localhost:7860/sdapi/v1/script-info. It's better to save this page as a
        # json file, and the use an online tool to beautify this json file for
        # better readibility

        reactor_args = [
            self._encode_file_to_base64(image_file),  # 0
            True,  # 1 Enable ReActor
            "0",  # 2 Comma separated face number(s) from swap-source image
            "0",  # 3 Comma separated face number(s) for target image (result)
            "inswapper_128.onnx",  # 4 model path
            "CodeFormer",  # 4 Restore Face: None; CodeFormer; GFPGAN
            1,  # 5 Restore visibility value
            True,  # 7 Restore face -> Upscale
            "4x_NMKD-Superscale-SP_178000_G",  # 8 Upscaler (type 'None' if
            # doesn't need), see full list here:
            # http://127.0.0.1:7860/sdapi/v1/script-info -> reactor -> sec.8
            1.5,  # 9 Upscaler scale value
            1,  # 10 Upscaler visibility (if scale = 1)
            False,  # 11 Swap in source image
            True,  # 12 Swap in generated image
            1,  # 13 Console Log Level (0 - min, 1 - med or 2 - max)
            0,  # 14 Gender Detection (Source) (0 - No, 1 - Female Only, 2 - Male Only)
            0,  # 15 Gender Detection (Target) (0 - No, 1 - Female Only, 2 - Male Only)
            False,  # 16 Save the original image(s) made before swapping
            0.8,  # 17 CodeFormer Weight (0 = maximum effect, 1 = minimum
            # effect), 0.5 - by default
            False,  # 18 Source Image Hash Check, True - by default
            False,  # 19 Target Image Hash Check, False - by default
            "CPU",  # 20 CPU or CUDA (if you have it), CPU - by default
            True,  # 21 Face Mask Correction
            0,  # 22 Select Source, 0 - Image, 1 - Face Model, 2 - Source Folder
            None,  # 23 Filename of the face model (from
            # "models/reactor/faces"), e.g. elena.safetensors, don't forger to
            # set #22 to 1
            None,  # 24 The path to the folder containing source faces images,
            # don't forger to set #22 to 2
            None,  # 25: Multiple Source Images skip it for API
            True,  # 26 Randomly select an image from the path
            True,  # 27 Force Upscale even if no face found
            0.6,  # 28 Face Detection Threshold
            2,  # 29 Maximum number of faces to detect (0 is unlimited)
        ]
        payload = {
            "prompt": prompt,  # extra networks also in prompts
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "n_iter": 1,
            "batch_size": batch_size,
            # example args for Refiner and ControlNet
            "alwayson_scripts": {"reactor": {"args": reactor_args}},
            "override_settings": {
                "sd_model_checkpoint": sd_model_checkpoint,
            },
        }
        self._call_txt2img_api(**payload)

    def remake_image(
        self,
        image_file="extensions/sd-webui-reactor/example/IamSFW.jpg",
        prompt="white hair, blue eyes",
        negative_prompt="3d, cartoon, anime, sketches, (worst quality, bad quality, child, cropped:1.4) (monochrome)), (grayscale)), (bad-hands-5:1.0), (badhandv4:1.0), (EasyNegative:0.8), (bad-artist-anime:0.8), (bad-artist:0.8), (bad_prompt:0.8), (bad-picture-chill-75v:0.8), (bad_prompt_version2:0.8), (bad_quality:0.8)",
        seed=-1,
        steps=30,
        width=512,
        height=512,
        denoising_strength=0.75,
        cfg_scale=7,
        sampler_name="DPM++ 2M",
        scheduler="karras",  # option exponential
        n_iter=1,
        batch_size=1,
        sd_model_checkpoint="revAnimated_v122EOL.safetensors",  # start from there for controlnet
        control_mode="Balanced",
        guidance_start=0,
        guidance_end=1,
        resize_mode="Crop and Resize",
        module="lineart_standard",
        model="control_v11p_sd15_lineart [43d4be0d]",
    ):

        # To get a list of model list for ControlNet, run this command:
        # curl -X 'GET' \
        # 'https://ac50f1a52aaefcd9ac.gradio.live/controlnet/model_list?update=true' \
        # -H 'accept: application/json'

        # Similarly, for module list:
        # curl -X 'GET' \
        # 'https://ac50f1a52aaefcd9ac.gradio.live/controlnet/module_list?alias_names=false' \
        # -H 'accept: application/json'

        # Frist, we have to download models

        payload = {
            "prompt": prompt,  # extra networks also in prompts
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": width,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "n_iter": 1,
            "batch_size": batch_size,
            # example args for Refiner and ControlNet
            "alwayson_scripts": {
                "ControlNet": {
                    "args": [
                        {
                            "batch_images": "",
                            "control_mode": control_mode,
                            "enabled": True,
                            "guidance_end": guidance_end,
                            "guidance_start": guidance_start,
                            "image": {
                                "image": self._encode_file_to_base64(image_file),
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
        self._call_txt2img_api(**payload)
