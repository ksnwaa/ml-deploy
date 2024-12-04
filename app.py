from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

#load model dan label encoder
model = load_model('disease_model.h5')
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

#features (symptoms)
with open('features.pkl', 'rb') as file:
    symptoms = pickle.load(file)

#mapping penyakit ke kode
with open('disease_code_map.pkl', 'rb') as file:
    disease_code_map = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_results = None
    selected_values = {f'symptom{i}': '' for i in range(1, 6)}
    
    if request.method == 'POST':
        # Simpan nilai yang dipilih
        selected_values = {
            f'symptom{i}': request.form[f'symptom{i}']
            for i in range(1, 6)
        }
        
        # Buat array kosong untuk input
        input_data = np.zeros(len(symptoms))
        
        # Set nilai 1 untuk symptoms yang dipilih
        selected_symptoms = [value for value in selected_values.values()]
        
        for symptom in selected_symptoms:
            if symptom in symptoms:
                idx = symptoms.index(symptom)
                input_data[idx] = 1
        
        # Reshape input untuk prediction
        input_data = input_data.reshape(1, -1)
        
        # Probabilitas dari prediksi
        predictions = model.predict(input_data)[0]
        
        # Top 3 prediksi
        top_3_idx = predictions.argsort()[-3:][::-1]
        prediction_results = []
        
        for idx in top_3_idx:
            disease = label_encoder.inverse_transform([idx])[0]
            probability = predictions[idx] * 100
            code = disease_code_map.get(disease, "N/A")
            prediction_results.append({
                'disease': disease,
                'code': code,
                'probability': probability
            })
    
    return render_template('index.html',
                         symptoms=symptoms,
                         prediction_results=prediction_results,
                         selected_values=selected_values)

if __name__ == '__main__':
    app.run(debug=True)
