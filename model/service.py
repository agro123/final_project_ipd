from flask import Flask, request, jsonify, send_file
import pandas as pd
import mlflow.sklearn
import time
import os

from pre_process import pre_process

app = Flask(__name__)

model_uri = "runs:/movies_revenue_model/random_forest_model" 
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    timestamp = time.time()
    try:
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
        result_file_path = f"./data/results/result_{timestamp}.csv" 
        result.to_csv(result_file_path, index=False)

        return send_file(result_file_path, as_attachment=True, download_name=f"result_{timestamp}.csv")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
