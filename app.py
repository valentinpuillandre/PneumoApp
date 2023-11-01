from __future__ import division, print_function
# coding=utf-8
import numpy as np
from PIL import Image
from io import BytesIO
# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

MODEL_PATH = 'models/trained_model_CNN_new.h5'

#Charger le dataset
model = load_model(MODEL_PATH)

#=======================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def predict_pneumonia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Aucun fichier n'a été soumis."
        
        file = request.files['file']
        
        if file.filename == '':
            return "Le nom du fichier est vide."

        if file and allowed_file(file.filename):
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes))
            image = image.convert("L")
            image = image.resize((150, 150))
            img_array = np.array(image) / 255
            img_array = img_array.reshape(-1, 150, 150, 1)
            prediction = model.predict(img_array)
            is_pneumonia = "Normal" if prediction[0] > 0.5 else "Pneumonie"
            return render_template('index.html', prediction=is_pneumonia)

    return render_template('index.html')



@app.route('/', methods=['GET'])
def index():
    title = 'Test page'
    # Page principal
    return render_template('index.html')

if __name__ == '__main__':
        app.run(debug=True)
    
#, port=int(os.environ.get("PORT", 8080))