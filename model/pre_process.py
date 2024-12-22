import pandas as pd
from sklearn.preprocessing import LabelEncoder

def pre_process(raw_data, include_revenue=False): 
    data = raw_data
    # Seleccionar columnas relevantes
    columns = [
        'genres', 'language', 'budget_usd',
        'vote_count', 'runtime_hour', 'runtime_min', 'director', 'user_score'
    ]
    if include_revenue:
        columns.append('revenue_usd')
    
    # Transformar columnas 'budget_usd', 'revenue_usd' en numericas
    data['budget_usd'] = pd.to_numeric(data['budget_usd'], errors='coerce')
    if include_revenue:
        data['revenue_usd'] = pd.to_numeric(data['revenue_usd'], errors='coerce')

    data = data[columns]
    
    # Eliminar filas con valores nulos
    data = data.dropna()
    
    # Codificar variables categóricas (Label Encoding para este ejemplo)
    le_language = LabelEncoder()
    data['language_encoded'] = le_language.fit_transform(data['language'])
    le_director = LabelEncoder()
    data['director_encoded'] = le_director.fit_transform(data['director'])
    data
    
    # Convertir géneros en columnas booleanas
    available_genders = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western', 'Short']
    genres_dummies = data['genres'].str.get_dummies(sep=',')
    # Fusionar columnas duplicadas de generos
    # (Porque se crean columnas con espacio al principio ejemplo 'Action' y ' Action')
    genres_dummies.columns = genres_dummies.columns.str.strip() #Eliminar espacios
    genres_dummies = genres_dummies.groupby(genres_dummies.columns, axis=1).any()
    
    #add missing gender columns
    missing_columns = set(available_genders) - set(genres_dummies.columns)
    if missing_columns:
        genres_dummies = genres_dummies.assign(**{col: False for col in missing_columns})
    
    data = pd.concat([data, genres_dummies], axis=1)
    data = data.drop(columns=['genres', 'language', 'director'])

    # Ordenar las columnas alfabéticamente
    data = data.reindex(sorted(data.columns), axis=1)
    return data