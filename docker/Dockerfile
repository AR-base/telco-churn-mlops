# Churn risk desk: the MLflow champion + prediction API + Streamlit website in one image.
# Build:  docker build -t churn-risk-desk:1.0 .
# Run:    docker run -d --name churn-desk -p 8501:8501 -p 8000:8000 churn-risk-desk:1.0
FROM python:3.13-slim

LABEL org.opencontainers.image.title="Churn risk desk" \
      org.opencontainers.image.description="Telco churn model (MLflow champion) with a prediction API and a Streamlit website" \
      org.opencontainers.image.authors="Pinninti Sai Anish, Paravasthu Vamsi Krishna Suri, Kola Karan" \
      org.opencontainers.image.version="1.0" \
      ml.model.name="telco-churn-classifier" \
      ml.model.version="1" \
      ml.model.config="LogReg-B_compact" \
      ml.model.mlflow_run_id="4f2d5400ce8244e4a7a1957e988670ac" \
      ml.model.sha256="61a402daca0bb06b034e8e562b7698570be0ccefcc0dff1d0762dc12f6208334"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/model \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1) dependencies first, so editing the app does not reinstall them (versions pinned to the training session)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) application + model (owned by root, read-only for the app user)
COPY app/ ./

# 3) refuse to build an image whose model does not reproduce the notebook's scores; add a non-root user
RUN python selftest.py && useradd --create-home --uid 10001 --user-group app
USER 10001:10001

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4); u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

CMD ["python", "start.py"]
