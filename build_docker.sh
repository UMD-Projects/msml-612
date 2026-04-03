# Replace with your project ID and desired image name/tag
export PROJECT_ID=$(gcloud config get-value project)
export REPO_NAME=flaxdiff-docker-repo
export IMAGE_NAME=flaxdiff-tpu-trainer
export IMAGE_TAG=latest
export REGION=europe-west4

export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

docker build -t $IMAGE_URI .
docker push $IMAGE_URI