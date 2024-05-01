from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests

model = load_model('model.h5')

app = Flask(__name__)

with open('data/list.txt', 'r') as file:
    animals = file.read().splitlines()

trans = "https://libretranslate.com/translate"

def classify(path):
    img = load_img(path, target_size=(200, 200))
    img = img_to_array(img)
    img = img.reshape(1, 200, 200, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    result = model.predict(img)
    index = np.argmax(result)
    animal = animals[index]
    return animal


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='Nevybrán žádný soubor!')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='Nevybrán žádný soubor!')
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            result = classify(file_path)
            return render_template('index.html', message='Obrázek nahrán!', prediction=result)
    return render_template('index.html', message='', prediction='')

if __name__ == '__main__':
    app.run(debug=True)
