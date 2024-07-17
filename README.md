# Stable Diffusion web UI & API in Docker
Stable Diffusion Automatic1111 implemented in Docker and docker compose for fast server deployment
![](screenshot.png)

## Usage
First, make sure you have [Docker](https://docs.docker.com/engine/install/) installed in your Linux server. In addition, you should also installed [docker compose](https://docs.docker.com/compose/install/) for quick and easy volumne attach, deployment, SSL, nginx and domain name.


### Install
First, clone this repo:
```bash
git clone --recursive https://github.com/vule20/stable-diffusion-webui.git
cd stable-diffusion-webui
```
#### Use with Docker

If you just wanna build docker image locally with Docker and don't wanna use docker-compose, simply run:
```bash
docker build -t stable-diffusion-api:latest .
```

Then, run this image, and mount it with the `data`, `models\Stable-diffusion`, `models/Lora`, `configs` and `outputs` folder as bellow:

```bash
docker run --rm --gpus all -it \
 -v models:/stable-diffusion/models \
 -v ./configs/:/stable-diffusion/configs \
 -v /outputs:/stable-diffusion/outputs \
 stable-diffusion-api:latest \
 python3 /stable-diffusion/launch.py --listen --api
```

#### Use with docker compose
If you prefer, you can use my `docker compose` version, update your domain name, and add new SSL cerf for your domain record. Simply run:
```bash
docker compose up -d
```

### Install new extensions, use with new models and Lora
To run this docker image with different extensions, simply clone the extentions you wanna use to the `extensions` folder:
```bash
git submodule add extension-url extensions/extension-name
```
For example:
```bash
git submodule add git submodule add https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor
```

To use different models, and Lora with this docker image, download your corresponding stable diffusion and Lora to `./models/Stable-diffusion` and `./models/Lora`.

The results folder when you run the docker container are located in `outputs`

### Download models
You can edit the `get_models.sh` to download models every time you deploy on a new server. In addition, you can also use the `download.py` script to download models from civit.ai. Simply run:
```bash
./download.py civit_ai_https_url local_folder
```

### Python API

After that, you can test the backend server by running the sample [python api scripts](./python_scripts/api.py). However, you should keep in mind that, the first run may be slow because the server needs to download some more dependecies such as missing models. Make sure you update the URL endpoint and other parameters such as models, Lora, prompt, and image

```bash
python3 api_example.py
```

## Demo with FaceSwap extension for Age Sliding 
https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/af2c383a-dfae-4afe-8eb2-bf701d96cec9/width=1800,quality=90/xyz_grid-0011-4193570023.jpeg

First, download this [Lora](https://civitai.com/models/128417?modelVersionId=143150) to `models/Lora`. By default, it will download Stable-Diffusion1.5 to `./models/Stable-diffusion`. If you wanna use another base model such as [Realistic Vision V6.0 B1](https://civitai.com/models/4201/realistic-vision-v60-b1?modelVersionId=125411), download and put it to `./models/Stable-diffusion`.


prompt: `asian man, white shirt, grey jacket, black hair, portrait, looking at viewer, forest, hat<lora:age_slider_v20:-1>`
negative prompt: `nude, breasts, topless, cartoon, cgi, render, illustration, painting, drawing, bad quality, grainy, low resolution`
Other metadata: Guidance=7, steps=20, sampler=`DPM++ 2M SDE Karras`
Lora: Age Slider
Base model: can be Photon, SDXL, Realistic Vision V6.0 B1

If you wanna change age, the working range (age_slider_range) for the Lora is from -5 to +5, this parameter is placed after the `<lora:age_slider_v20:age_slider_range>`. -5 is the youngest (`<lora:age_slider_v20:-5>`), and 5 is the oldest (`<lora:age_slider_v20:5>`).

# Example Usage
## Anieme - Style Conversion
negative prompt:
```bash
3d, cartoon, anime, sketches, (worst quality, bad quality, child, cropped:1.4) (monochrome)), (grayscale)), (bad-hands-5:1.0), (badhandv4:1.0), (EasyNegative:0.8), (bad-artist-anime:0.8), (bad-artist:0.8), (bad_prompt:0.8), (bad-picture-chill-75v:0.8), (bad_prompt_version2:0.8), (bad_quality:0.8)
```

## Contacts
Vu Le, University of Massachusetts Amherst at vdle@umass.edu