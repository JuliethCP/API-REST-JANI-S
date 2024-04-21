from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime
import pandas as pd  # Necesario para convertir las fechas en SARIMAX
import statsmodels.api as sm
from flask_cors import CORS

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
