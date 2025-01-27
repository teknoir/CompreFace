# Teknoir Changes

The repo contain some minimal config changes to be able to ebuild and run CompreFace in the Teknoir platform.

## Add models

Open the file `embedding-calculator/Dockerfile` go to line 71, and add your models to the `/app/ml/.models` dir something like this.
```Dockerfile
COPY <path to my model> /app/ml/.models/
```

## Custom build

Navigate to `dev` and run:
```bash
docker compose build
docker compose push
```

The custom build now creates the following images:
* us-docker.pkg.dev/teknoir/gcr.io/compreface-core:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-admin:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-api:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-postgres-db:latest
* us-docker.pkg.dev/teknoir/gcr.io/compreface-fe:latest

## Deploy
Our own Helm Chart for Compreface can be found here:
https://github.com/teknoir/compreface-helm


Example deployment:
On the Deploy tab look for the CompreFace node:
https://teknoir.cloud/devstudio/teknoir-ai/transformers/#flow/efd9e54cb8a64f6f

Navigate to:
https://compreface-trainer-2x-rtx3090-teknoir-ai.teknoir.cloud/login