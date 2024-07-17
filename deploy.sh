#/bin/bash

echo "Installing nvidia-container-toolkit"
sudo apt-get install -y nvidia-container-toolkit

echo "Dowloading models and Lora. If you wanna add more models, just put links in the scripts/model_downloader.py file and then re-run the deploy.sh command or you can just run python3 scripts/model_downloader.py seperately."

python3 scripts/model_downloader.py

echo "running docker compose up for deploy"
docker compose up --force-recreate -d
