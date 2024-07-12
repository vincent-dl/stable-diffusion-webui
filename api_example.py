from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = "http://localhost:7860"

out_dir = "outputs"
out_dir_t2i = os.path.join(out_dir, "txt2img")
os.makedirs(out_dir_t2i, exist_ok=True)


