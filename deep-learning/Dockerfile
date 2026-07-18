FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow==3.14.0"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", "--serve-artifacts", "--artifacts-destination", "/mlflow-data/artifacts"]
