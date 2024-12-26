FROM jupyter/datascience-notebook:latest

WORKDIR /app

COPY /model/* /app/model/


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r model/requirements.txt

#CMD ["sh", "-c", "python model/model.py & mlflow ui --host 0.0.0.0 --port 5000"]