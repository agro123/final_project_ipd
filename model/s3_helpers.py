import boto3
import os

s3 = boto3.client('s3')
BUCKET_NAME = os.getenv("S3_BUCKET")

# Función para descargar archivos desde S3
def download_from_s3(key, local_path):
    s3.download_file(BUCKET_NAME, key, local_path)

# Función para subir archivos a S3
def upload_to_s3(key, local_path):
    s3.upload_file(local_path, BUCKET_NAME, key)
