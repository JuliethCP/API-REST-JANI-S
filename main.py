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
    data = request.get_json()
    variable_name = data['variable_name']  # Variable enviada desde el frontend
    variable_value = data['variable_value']  # Valor de la variable enviada desde el frontend

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
    else:
        # Convierte el valor de la variable a un array bidimensional de una sola fila
        variable_value = np.array([[1, float(variable_value)]])
        # Verifica si hay un modelo asociado a la variable
        model_path = os.path.join(current_dir, 'models', f'{variable_name}_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'Modelo para la variable {variable_name} no encontrado'}), 404

        model = joblib.load(model_path)

        try:
            prediction = model.predict(variable_value)
            return jsonify({'prediction': prediction[0]})
        except Exception as e:
            return jsonify({'error': f'Error al hacer la predicción con el modelo {variable_name}: {e}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
