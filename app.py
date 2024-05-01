from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('model.h5')

app = Flask(__name__)

def classify_image(image_path):
    img = load_img(image_path, target_size=(200, 200))
    img = img_to_array(img)
    img = img.reshape(1, 200, 200, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    result = model.predict(img)
    return result[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            result = classify_image(file_path)
            if result > 0.5:
                prediction = 'Je to pes!'
            else:
                prediction = 'Je to kočka!'
            return render_template('index.html', message='Obrázek nahrán!', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
