# Teknoir Changes

The repo contain some minimal config changes to be able to build and run CompreFace in the Teknoir platform.
Changes has been made for custom builds and build scripts to 

## Custom builds

Navigate to `dev` dir and run:
```bash
docker compose build
docker compose push
```

Build additional GPU images, navigate to `embedding-calculator` dir and run:
```bash
make build-images
docker push --all-tags us-docker.pkg.dev/teknoir/gcr.io/compreface-core-base
docker push --all-tags us-docker.pkg.dev/teknoir/gcr.io/compreface-core
```

The custom build now creates the following images:
* us-docker.pkg.dev/teknoir/gcr.io/compreface-admin:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-api:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-postgres-db:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-fe:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest-mobilenet
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest-facenet
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest-arcface-r100
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest-mobilenet-gpu
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest-arcface-r100-gpu

# Add your own model

## Example custom gpu accelerated facenet model
There were no build scripts for gpu accelerated facenet images in CompreFace, so we added some.

See [Dockerfile](teknoir/Dockerfile) how a facenet model replace the existing model just to be a carrier to be copied 
into the actual image at build time.

See [Makefile](embedding-calculator/Makefile) have been updated with a custom facenet model build.

Nvigate to the `embedding-calculator` dir and manually build the new images: 
```bash
make build-cuda
make build-custom-facenet-gpu
docker push us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest-custom-facenet-gpu
```

## Deploy
Our own Helm Chart for CompreFace can be found here:
https://github.com/teknoir/compreface-helm

Example deployment:
On the Deploy tab look for the CompreFace node:
https://teknoir.cloud/devstudio/teknoir-ai/transformers/#flow/efd9e54cb8a64f6f

Navigate to:
https://compreface-trainer-2x-rtx3090-teknoir-ai.teknoir.cloud/login