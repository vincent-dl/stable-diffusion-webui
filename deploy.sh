#/bin/bash

DOCKERFILE=Dockerfile
IMAGE_NAME=stable-diffusion-api:latest

if [[ $DOCKERFILE -nt "$(docker inspect -f '{{.Created}}' $IMAGE_NAME)" ]]; then
    echo "Dockerfile has changed. Rebuilding $IMAGE_NAME..."
    docker build -t $IMAGE_NAME .
else
    echo "Dockerfile has not changed. Skipping rebuild."
fi

echo "Dowloading models from get_models.sh. If you wanna add more models, just put links in file and then re-run the deploy.sh command or you can just run sh get_models.sh seperately."
sh get_models.sh

echo "running docker compose up for deploy"
docker compose up
