from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime
import pandas as pd  # Necesario para convertir las fechas en SARIMAX
import statsmodels.api as sm
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(variable_name):
    model_filename = f"{variable_name}_model.pkl"
    model_path = os.path.join(current_dir, 'models', model_filename)
    try:
        # Carga el modelo desde el archivo
        model = sm.load(model_path)
        return model
    except Exception as e:
        print(f"Error al cargar el modelo de {variable_name}: {e}")
        return None

def predictS(variable_name, input_date):
    model = load_model(variable_name)
    if model is None:
        return None

    # Realiza la predicción para la fecha dada utilizando el método SARIMAX
    prediction = process_SARIMAX(model, input_date)
    return prediction

def predict_house_model(taxvaluedollarcnt, taxamount):
    # Carga el modelo house
    model_path = os.path.join(current_dir, 'models', 'house_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Modelo para la variable house no encontrado'}), 404

    # Carga el modelo asociado a la variable
    model = joblib.load(model_path)

    # Realiza la predicción
    prediction = model.predict([[taxvaluedollarcnt, taxamount]])

    return prediction[0]

def process_SARIMAX(model, input_date):
    try:
        # Convertir la fecha de entrada a un objeto datetime
        input_datetime = pd.to_datetime(input_date)
        pred = model.get_prediction(start=input_datetime, end=input_datetime, dynamic=False)
        predicted_value = pred.predicted_mean[input_datetime]
        return predicted_value
    except Exception as e:
        print(f"Error predicting with SARIMAX model: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Obtén los datos de la solicitud
    data = request.get_json()
    variable_name = data['variable_name']  # Variable enviada desde el frontend
    variable_value = data['variable_value']  # Valor de la variable enviada desde el frontend

    # Si la variable es una de las que requiere un timestamp
    if variable_name in ['avocado', 'bitcoin', 'SP500Stock']:
        if isinstance(variable_value, str):
            variable_value = datetime.strptime(variable_value, '%Y-%m-%d')
        else:
            return jsonify({'error': 'Se esperaba un timestamp para esta variable'}), 400

        prediction = predictS(variable_name, variable_value)

        if prediction is not None:
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Error al hacer la predicción'}), 500
    elif variable_name == 'auto':
        #este recibe 2 variables año y kilometrso recorridos (ej 2022 y 500000 )
        variable_value = data['variable_value']
        


    elif variable_name == 'house':
        variable_value = data['variable_value']
        # Asegúrate de que las dos variables necesarias estén presentes en la solicitud
        if 'taxvaluedollarcnt' not in variable_value or 'taxamount' not in variable_value:
            return jsonify({'error': 'Se requieren taxvaluedollarcnt y taxamount para la variable house'}), 400

        taxvaluedollarcnt = variable_value['taxvaluedollarcnt']
        taxamount = variable_value['taxamount']

        # Realiza la predicción
        prediction = predict_house_model(taxvaluedollarcnt, taxamount)

        return jsonify({'prediction': prediction})
    elif variable_name == 'movie':
        # Obtener el ID del usuario de la solicitud
        userId = variable_value

        model_path = os.path.join(current_dir, 'models', f'movie_recommendation_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'Modelo para la variable {variable_name} no encontrado'}), 404

        # Carga el modelo asociado a la variable
        model = joblib.load(model_path)

        if model is None:
            return jsonify({'error': 'Error al cargar el modelo de recomendación de películas'}), 500

        # Realizar predicciones para el usuario especificado
        predictions = []
        for movieId in range(1, 149532):  # Rango de IDs de películas, ajusta según tus datos
            prediction = model.predict(userId, movieId)
            predictions.append((movieId, prediction.est))

        # Ordenar las predicciones y obtener las 10 mejores
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:10]

        # Formatear las recomendaciones
        recommendations = []
        for movie in top_movies:
            movieId, estimated_rating = movie
            # Aquí podrías cargar los títulos de las películas desde un archivo CSV si lo deseas
            movie_title = f"Movie {movieId}"  # Cambia esto según tus datos
            recommendations.append({'movie_title': movie_title, 'estimated_rating': estimated_rating})

        # Devolver las recomendaciones
        return jsonify({'recommendations': recommendations})
        # Realiza la predicción dependiendo de la variable

    elif variable_name == 'stroke':

        # Verifica si hay un modelo asociado a la variable
        model_path = os.path.join(current_dir, 'models', f'{variable_name}_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'Modelo para la variable {variable_name} no encontrado'}), 404

        model = pickle.load(open(model_path, 'rb'))

        # Extrae los valores de las características del usuario del JSON enviado
        gender = float(variable_value['gender'])
        age = float(variable_value['age'])
        hypertension = float(variable_value['hypertension'])
        heart_disease = float(variable_value['heart_disease'])
        ever_married = float(variable_value['ever_married'])
        work_type = float(variable_value['work_type'])
        Residence_type = float(variable_value['Residence_type'])
        avg_glucose_level = float(variable_value['avg_glucose_level'])
        bmi = float(variable_value['bmi'])
        smoking_status = float(variable_value['smoking_status'])

        # Crea un numpy array con los datos de entrada del usuario
        user_input = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                                avg_glucose_level, bmi, smoking_status]])

        # Realiza la predicción de la probabilidad de accidente cerebrovascular
        stroke_probability = model.predict_proba(user_input)[0][1]

        # Devuelve el resultado de la predicción
        if stroke_probability > 0.5:
            prediction_result = {'stroke_risk': 'Alto', 'probability': stroke_probability}
        else:
            prediction_result = {'stroke_risk': 'Bajo', 'probability': stroke_probability}

        return jsonify(prediction_result)
    else:
        # Si no es una de esas variables, conviértela en un array bidimensional de una sola fila
        variable_value = np.array([[float(variable_value)]])

    # Verifica si hay un modelo asociado a la variable
    model_path = os.path.join(current_dir, 'models', f'{variable_name}_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': f'Modelo para la variable {variable_name} no encontrado'}), 404

    # Carga el modelo asociado a la variable
    model = joblib.load(model_path)

    # Realiza la predicción
    prediction = model.predict(variable_value)

    # Devuelve la predicción
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)