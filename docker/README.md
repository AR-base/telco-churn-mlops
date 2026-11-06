# Churn risk desk: Docker image

This folder builds one image that contains:

- the champion model exported from the MLflow registry (`telco-churn-classifier` version 1, LogReg-B_compact);
- the prediction API (FastAPI, port 8000);
- the website (Streamlit, port 8501).

Training happened in the notebook. The image only serves the model, so it needs no data and no MLflow.

## 1. Install Docker (once)

- **Windows or Mac:** install **Docker Desktop** from docker.com, start it, and wait until it shows *Engine running*. On Windows, accept the WSL 2 option during setup.
- Check it works in a terminal (PowerShell on Windows):

```
docker version
```

## 2. Build the image

In a terminal, go to this folder (the one with the `Dockerfile`):

```
cd path\to\churn-risk-desk-docker
docker build -t churn-risk-desk:1.0 .
```

The first build takes 3–5 minutes. It ends by running `selftest.py` inside the image, so look for **`4/4 checks passed`** in the output. If a check fails, the build stops: an image with a broken model is never created.

## 3. Run it on your laptop

```
docker run -d --name churn-desk -p 8501:8501 -p 8000:8000 --restart unless-stopped churn-risk-desk:1.0
```

- Website: <http://localhost:8501>
- API docs (try `/predict` in the browser): <http://localhost:8000/docs>
- After about 30 seconds, `docker ps` shows the container as `(healthy)`.
- See the logs with `docker logs churn-desk`.
- Stop with `docker stop churn-desk`, start again with `docker start churn-desk`, and delete with `docker rm -f churn-desk`.

If port 8501 is already used, map another one, e.g. `-p 9000:8501`, then open <http://localhost:9000>.

## 4. Share it

### Option A: Docker Hub (anyone can run it with one command)

1. Create a free account on hub.docker.com. Your username is what replaces `YOUR_USER` below.
2. Log in from the terminal:

```
docker login
```

3. Build and upload for both Windows/Intel and Apple-silicon (M1–M4) computers. The first command is needed only once.

```
docker buildx create --name multiarch --use
docker buildx build --platform linux/amd64,linux/arm64 -t YOUR_USER/churn-risk-desk:1.0 -t YOUR_USER/churn-risk-desk:latest --push .
```

This takes about 10 minutes, because the Apple version is built under emulation.

4. On hub.docker.com, open the repository and check that **Visibility** is **Public**.

Anyone with Docker can then run it, with nothing else to install:

```
docker run -p 8501:8501 YOUR_USER/churn-risk-desk:1.0
```

### Option B: a file (no account, e.g. for a USB stick, Drive or Teams)

```
docker save -o churn-risk-desk-1.0.tar churn-risk-desk:1.0
tar -czf churn-risk-desk-1.0.tar.gz churn-risk-desk-1.0.tar
```

The receiver runs:

```
docker load -i churn-risk-desk-1.0.tar.gz
docker run -p 8501:8501 churn-risk-desk:1.0
```

The file works on the same kind of computer it was built on. Option A covers both kinds.

## Which model is inside?

The image is labelled with the model's MLflow run ID and checksum:

```
docker inspect --format "{{json .Config.Labels}}" churn-risk-desk:1.0
```

To check it again at any time:

```
docker run --rm churn-risk-desk:1.0 python selftest.py
```

## After retraining

1. Run the notebook again.
2. Download the new zip from §13.
3. Build with a new tag, for example `churn-risk-desk:1.1`.
4. Push it.

Old tags keep working, so anyone can go back to an earlier model.

## Security

- The image contains no data and no passwords or keys.
- It runs as a normal user (not root), and its code is read-only.
- The model is saved with skops, not pickle, so loading it cannot run hidden code. Its checksum is verified at build time and at every start.
- The website has **no login**. Use it on your own machine or a trusted network. Don't put it on the open internet with real customer data.
