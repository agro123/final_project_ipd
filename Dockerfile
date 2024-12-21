FROM python:3.9-slim-buster

WORKDIR /app

COPY /model/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY /model/* .

EXPOSE 5000

CMD ["sh", "-c", "python model.py & mlflow ui --host 0.0.0.0 --port 5000"]