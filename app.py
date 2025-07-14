from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
import os

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Cargar todos los componentes
try:
    # Cargar modelo y componentes de preprocesamiento
    model = joblib.load('models/knn_model.pkl')
    pca = joblib.load('models/pca_transformer.pkl')
    scaler = joblib.load('models/standard_scaler.pkl')
    sex_encoder = joblib.load('models/sex_encoder.pkl')
    embarked_encoder = joblib.load('models/embarked_encoder.pkl')
    cabin_encoder = joblib.load('models/cabin_encoder.pkl')
    
    logging.info("✅ Todos los componentes cargados correctamente")
except Exception as e:
    logging.error(f"❌ Error al cargar componentes: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        form_data = request.form
        
        # Validar y convertir datos
        input_data = {
            'Pclass': int(form_data.get('pclass')),
            'Sex': form_data.get('sex'),
            'Age': float(form_data.get('age')),
            'Fare': float(form_data.get('fare')),
            'Cabin': form_data.get('cabin', 'U').upper()[0] if form_data.get('cabin') else 'U'  # Tomar solo la primera letra o 'U' si está vacío
        }
        
        # Validar valores
        if input_data['Age'] < 0 or input_data['Age'] > 100:
            raise ValueError("La edad debe estar entre 0 y 100 años")
            
        if input_data['Fare'] < 0:
            raise ValueError("La tarifa no puede ser negativa")
            
        if input_data['Pclass'] not in [1, 2, 3]:
            raise ValueError("La clase debe ser 1, 2 o 3")
            
        if input_data['Sex'] not in ['male', 'female']:
            raise ValueError("Sexo debe ser 'male' o 'female'")
            
        if input_data['Cabin'] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U']:
            raise ValueError("Cabina debe ser una letra entre A-G, T o U")

        # Crear DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocesamiento
        df['Sex'] = sex_encoder.transform(df[['Sex']])
        df['Cabin'] = cabin_encoder.transform(df[['Cabin']])
        
        # Seleccionar features importantes (en el mismo orden que se entrenó el modelo)
        features = ['Sex', 'Age', 'Fare', 'Pclass', 'Cabin']
        X = df[features]
        
        # Escalar
        X_scaled = scaler.transform(X)
        
        # Aplicar PCA
        X_pca = pca.transform(X_scaled)
        
        # Predecir
        prediction = model.predict(X_pca)[0]
        proba = model.predict_proba(X_pca)[0][1]  # Probabilidad de sobrevivir
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(proba)
        })
        
    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        })
    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Error en el servidor: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True)
