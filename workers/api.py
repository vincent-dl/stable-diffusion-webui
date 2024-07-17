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
        image_file="extensions/sample_images/Jensen-NVIDIA.png",
        age=20,
        lora_range=0.5,
        gender="man",
        prompt="asian man, white shirt, grey jacket, black hair, portrait, looking at viewer, forest",
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
        sd_model_checkpoint="realisticVisionV60B1_v60B1VAE.safetensors",  # start from there for controlnet
        control_mode="Balanced",
        guidance_start=0,
        guidance_end=1,
        resize_mode="Crop and Resize",
        module="ip-adapter_face_id_plus",
        model="ip-adapter-faceid-plus_sd15 [d86a490f]",
    ):
        # lora range should be from 0->1.

        prompt = (
            prompt
            + f"<lora:ip-adapter-faceid-plus_sd15_lora:{lora_range}>, portrait photo of a {age}yo {gender}"
        )
        negative_prompt = (
            negative_prompt
            + "cropped, frame, text, deformed, glitch, noise, noisy, off-center, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white, 3D render, comics, signature, words, bad proportions, blurry, extra arms, extra legs, jpeg artifacts, low quality, lowres, malformed limbs, mutilated, distorted, cloned face, dehydrated, error, fused fingers, gross proportions, long neck, missing arms, missing legs, morbid, mutation, out of frame, poorly drawn face, poorly drawn hands, too many fingers, username, watermark, worst quality, unprofessional, cluttered, unappealing, pixelated"
        )
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
            "n_iter": n_iter,
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
            "enable_hr": True,
            "denoising_strength": denoising_strength,
            "override_settings": {
                "sd_model_checkpoint": sd_model_checkpoint,
            },
        }
        self._call_txt2img_api(**payload)

    def remake_image(
        self,
        image_file="extensions/sample_images/Jensen-NVIDIA.png",
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
            "height": height,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "n_iter": n_iter,
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
            "enable_hr": True,
            "denoising_strength": denoising_strength,
            "override_settings": {
                "sd_model_checkpoint": sd_model_checkpoint,
            },
        }
        self._call_txt2img_api(**payload)
