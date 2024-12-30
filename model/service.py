from flask import Flask, request, jsonify, send_file
import pandas as pd
import mlflow.sklearn
from joblib import load

import tempfile
import logging
import time
import os

from pre_process import pre_process
from s3_helpers import download_from_s3, upload_to_s3

app = Flask(__name__)

VERSION = os.getenv("MODEL_VERSION", "v1")

print('---------------->', VERSION)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    timestamp = time.time()
    try:
        #Download model from S3
        model_key = f'models/revenue_prediction_{VERSION}.joblib'
        #local_model_path = f'./temp/revenue_prediction_{VERSION}.joblib'
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_model:
            download_from_s3(model_key, tmp_model.name)
            model = load(tmp_model.name)

        raw_data = pd.read_csv(file)

        #Validate columns
        necesary_columns = ['genres', 'language', 'budget_usd', 'vote_count', 'runtime_hour', 'runtime_min', 'director', 'user_score']

        is_valid_columns = all(col in raw_data.columns for col in necesary_columns)
        if(not is_valid_columns):
            raise Exception("Invalid format")

        pp_data = pre_process(raw_data)

        if pp_data.shape[0] == 0:
            raise Exception("Incomplete data")

        predictions = model.predict(pp_data) 
        result = raw_data 
        result['revenue_usd'] = predictions 
        result['revenue_usd'] = pd.to_numeric(result['revenue_usd'], errors='coerce')

        # Save the CSV file
        file_name = f'result_{VERSION}_{timestamp}.csv'
        #result_file_path = f"./temp/{file_name}"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_result:
            result_file_path = tmp_result.name
            result.to_csv(result_file_path, index=False)  

            result_key = f'results/{file_name}'
            upload_to_s3(result_key, result_file_path)

        return send_file(result_file_path, as_attachment=True, download_name=f"{file_name}.csv")

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction. Please try again later."}), 500

    finally:
            # clean temp data
            if os.path.exists(tmp_model.name):
                os.remove(tmp_model.name)
            if os.path.exists(result_file_path):
                os.remove(result_file_path)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
