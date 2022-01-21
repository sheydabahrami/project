# api_test.py
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

IMG_SHAPE = (224,224)
CLASSES = ["Modern", "Old"]
model = load_model('fine_tuned_house.h5')

app = Flask(__name__)

# basic first page
@app.route('/')
def intro():
      return render_template('index.html')

@app.route('/predict-interior', methods=['POST'])
def predict():
    f = request.files['img']
    file = Image.open(f)
    file_shape = np.asarray(file).shape
    #### PREDICTION STARTS HERE ####    
    preds = model.predict(np.expand_dims(file, axis=0))[0]
    i = np.argmax(preds)
    label = CLASSES[i]
    prob = preds[i]
    
    predictions={"label": label, "prob": str(prob)}
    return jsonify(predictions=predictions)