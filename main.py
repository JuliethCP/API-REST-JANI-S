from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime
import pandas as pd  # Necesario para convertir las fechas en SARIMAX
import statsmodels.api as sm
from flask_cors import CORS
import pickle
import base64
from io import BytesIO
from PIL import Image
import cv2
import tensorflow as tf

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
        
        
    elif variable_name == 'automobile':
        #este recibe 2 variables año y kilometrso recorridos (ej 2022 y 500000 )
        variable_value = data['variable_value']
        if 'year' not in variable_value or 'kilometers' not in variable_value:
            return jsonify({'error': 'Se requieren Year  y Kilometers para la variable auto'}), 400
        
        year = variable_value['year']
        kilometers = variable_value['kilometers']

        model_path = os.path.join(current_dir, 'models', 'automobile_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Modelo para la variable auto no encontrado'}), 404

         # Carga el modelo asociado a la variable
        model = joblib.load(model_path)

         # Realiza la predicción
         
    # Realiza la predicción
        prediction = model.predict([[year, kilometers]])

        return jsonify({'prediction': prediction[0]})
    
    elif variable_name == 'crimes':
        #este recibe 2 variables año y kilometrso recorridos (ej 2022 y 500000 )
        variable_value = data['variable_value']
        if 'year' not in variable_value or 'month' not in variable_value:
            return jsonify({'error': 'Se requieren year  y month para la variable crimes'}), 400
        
        year = variable_value['year']
        month = variable_value['month']

        model_path = os.path.join(current_dir, 'models', 'crimes_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Modelo para la variable house no encontrado'}), 404

         # Carga el modelo asociado a la variable
        model = joblib.load(model_path)

         # Realiza la predicción
        prediction = model.predict([[year, month]])

        return jsonify({'prediction': prediction[0]})
    
    elif variable_name == 'covid':
        #este recibe 2 variables año y kilometrso recorridos (ej 2022 y 500000 )
        variable_value = data['variable_value']
        if 'confirmed' not in variable_value or 'deaths' not in variable_value:
            return jsonify({'error': 'Se requieren Confirmed  y Deaths para la variable covid'}), 400
        
        confirmed = variable_value['confirmed']
        deaths = variable_value['deaths']

        model_path = os.path.join(current_dir, 'models', 'covid_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Modelo para la variable house no encontrado'}), 404

         # Carga el modelo asociado a la variable
        model = joblib.load(model_path)

         # Realiza la predicción
        prediction = model.predict([[confirmed, deaths]])

        return jsonify({'prediction': prediction[0]})


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
    
    elif variable_name == 'company':

        # Verifica si hay un modelo asociado a la variable
        model_path = os.path.join(current_dir, 'models', f'{variable_name}_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'Modelo para la variable {variable_name} no encontrado'}), 404

        model = pickle.load(open(model_path, 'rb'))

        # Extrae los valores de las variables de entrada del diccionario
        tenure = float(variable_value['tenure'])
        monthly_charges = float(variable_value['MonthlyCharges'])
        contract = float(variable_value['Contract'])
        internet_service = float(variable_value['InternetService'])
        tech_support = float(variable_value['TechSupport'])

        # Crea un numpy array con los datos de entrada del usuario
        user_input = np.array([[tenure, monthly_charges, contract, internet_service, tech_support]])

        # Realiza la predicción del modelo
        company_probability = model.predict_proba(user_input)[0][1]

    # Devuelve el resultado de la predicción
        if company_probability > 0.6:
        
            prediction_result = {'prediction' : company_probability}
        else:
            prediction_result = {'prediction': company_probability}

        return jsonify(prediction_result)

   
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

def preprocess_image(image, target_size=(48, 48)):
    """
    Preprocesses an image for emotion recognition.

    This function performs the following preprocessing steps:
    1. Converts the image to grayscale.
    2. Resizes the image to the target size.
    3. Normalizes pixel values to the range [0, 1].
    4. Converts the image to a tensor and expands its dimensions to match the input shape of the model.

    Args:
        image (numpy.ndarray): An image array of shape (height, width, channels).
        target_size (tuple of int, optional): The target size for resizing the image. Defaults to (48, 48).

    Returns:
        tensorflow.Tensor: A tensor of shape (1, target_size[0], target_size[1], 1) containing the preprocessed image.
    """
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    resized_image = cv2.resize(grayscale_image, target_size)

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Convert image to a tensor and expand dimensions to match the input shape of the model
    expanded_image = tf.expand_dims(tf.convert_to_tensor(normalized_image, dtype=tf.float32), axis=-1)
    expanded_image = tf.expand_dims(expanded_image, axis=0)  # Add batch dimension

    return expanded_image


@app.route('/emotion', methods=['POST'])
def emotion():
    try:
        # Obtén los datos de la solicitud
        data = request.get_json()
        image_base64 = data['image']

        # Decodifica la imagen base64
        image_data = base64.b64decode(image_base64.split(',')[1])

        # Convertir a imagen usando PIL
        image = Image.open(BytesIO(image_data))

        # Convertir imagen PIL a numpy array
        image = np.array(image)

        # Preprocesar la imagen
        processed_image = preprocess_image(image)

        # Cargar el modelo (en formato .h5)
        model_path = os.path.join('emotions.h5')
        model = tf.keras.models.load_model(model_path)

        # Hacer la predicción
        predictions = model.predict(processed_image)
        predicted_emotion = np.argmax(predictions)

        # Mapear el índice de la predicción a una etiqueta de emoción
        label_mapping = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
        emotion_label = label_mapping[predicted_emotion]

        return jsonify({"emotion": emotion_label})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process image"}), 500
    
    
@app.route('/weapons', methods=['POST'])
def weapons():
    try:
        class_names = [
            'Arma de fuego corta',
            'Arma de fuego larga',
            'Munición',
            'Explosivo',
            'Cuchillo',
            'Palo',
            'Piedra',
            'Botella',
            'Otro',
            'Ninguno'
        ]
        # Load the model with custom object handling
        model_path = os.path.join('weapons.h5')
        custom_objects = {'GlorotUniform': tf.keras.initializers.glorot_uniform}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        # Get the data from the request
        data = request.get_json()
        image_base64 = data['image']

        # Decode the base64 image
        image_data = base64.b64decode(image_base64.split(',')[1])

        # Convert to image using PIL
        image = Image.open(BytesIO(image_data))

        # Preprocess the image for the model
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction
        predictions = model.predict(image_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        # Get the class name
        class_name = class_names[predicted_class]

        # Count the number of objects of the predicted class (assumed to be 1)
        object_count = 1

        return jsonify({"class": class_name, "count": object_count})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process image"}), 500
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)