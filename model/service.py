from flask import Flask, request, jsonify
import pandas as pd
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlflow.models import validate_serving_input

app = Flask(__name__)

model_uri = "runs:/movie_revenue_model/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        rawData = pd.read_csv(file)
        print('Raw data shape', rawData.shape)

        #Validate columns
        columns = ['genres', 'language', 'budget_usd', 'vote_count', 'runtime_hour', 'runtime_min', 'director', 'user_score']

        hasAllComuns = all(col in rawData.columns for col in columns)
        if(not hasAllComuns):
            raise Exception("Invalid format")
        #clean values
        preData = rawData[columns].dropna()

        # Parse into numeric columnns
        preData['budget_usd'] = pd.to_numeric(preData['budget_usd'], errors='coerce')

        #Codify columns
        le_language = LabelEncoder()
        preData['language_encoded'] = le_language.fit_transform(preData['language'])

        le_director = LabelEncoder()
        preData['director_encoded'] = le_director.fit_transform(preData['director'])

        #Scale
        scaler = StandardScaler()
        preData[['budget_usd', 'vote_count', 'runtime_hour', 'runtime_min']] = scaler.fit_transform(
            preData[['budget_usd', 'vote_count', 'runtime_hour', 'runtime_min']]
        )

        availableGenders = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

        dataGenders = preData['genres'].str.get_dummies(sep=',')
        dataGenders.columns = dataGenders.columns.str.strip()
        dataGenders = dataGenders.groupby(dataGenders.columns, axis=1).any()

        #add missing gender columns
        missing_columns = set(availableGenders) - set(dataGenders.columns)
        if missing_columns:
                dataGenders = dataGenders.assign(**{col: False for col in missing_columns})
        
        preData = pd.concat([preData, dataGenders], axis=1).drop(columns=['genres'])

        preData = preData.drop(columns=['language', 'director'])

        predictions = model.predict(preData)

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
