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
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['POST'])
def predict_pneumonia():
    if 'file' not in request.files:
        return "Aucun fichier n'a été soumis."
    file = request.files['file']
    if not file:
        return "Le fichier n'a pas été soumis."

    if file.filename == '':
        return "Le nom du fichier est vide."

    # Read the image from POST data
    image_bytes = file.read()
    image = Image.open(BytesIO(image_bytes))

    # Convert the image to grayscale
    image = image.convert("L")

    # Resize the image
    image = image.resize((150, 150))

    # Convert the image to a NumPy array
    img_array = np.array(image) / 255
    img_array = img_array.reshape(-1, 150, 150, 1)

    # Make a prediction with the model
    prediction = model.predict(img_array)
    is_pneumonia = "Normal" if prediction > 0.5 else "Pneumonie"
    return render_template('index.html', prediction=is_pneumonia)


# @app.route('/', methods=['GET', 'POST'])
# def predict_pneumonia():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             # Charger et prétraiter l'image
#             img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (150, 150))
#             img = np.array(img) / 255
#             img = img.reshape(-1, 150, 150, 1)

#             # Faire une prédiction avec le modèle
#             prediction = model.predict(img)
#             is_pneumonia = "Normal" if prediction > 0.5 else "Pneumonie"

#             return render_template('index.html', filename=filename, prediction=is_pneumonia)

#     return render_template('index.html')
#============================================================================


@app.route('/', methods=['GET'])
def index():
    title = 'Test page'
    # Page principal
    return render_template('index.html')

if __name__ == '__main__':
        app.run(debug=True, host="0.0.0.0")
    
#, port=int(os.environ.get("PORT", 8080))