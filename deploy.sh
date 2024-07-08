#/bin/bash

echo "Installing nvidia-container-toolkit"
sudo apt-get install -y nvidia-container-toolkit

echo "Dowloading models from get_models.sh. If you wanna add more models, just put links in file and then re-run the deploy.sh command or you can just run sh get_models.sh seperately."

sh get_models.sh

echo "running docker compose up for deploy"
docker compose up --force-recreate -d
