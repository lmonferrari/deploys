import os
from flask import Flask, render_template, request
import base64
from jsons.load import init
import numpy as np
from io import BytesIO
from PIL import Image

IMAGE_SIZE = (28, 28)
diretorio_atual = os.getcwd()
diretorio_modelo = 'modelo'
diretorio_json = 'jsons'
diretorio_pesos = 'modelo'

app = Flask(__name__)
model = init(os.path.join(diretorio_atual, diretorio_json, 'modelo.json'),
             os.path.join(diretorio_atual, diretorio_pesos, 'modelo1.h5'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    _, encode = str(imgData).split(',', 1)
    decoded = base64.b64decode(encode)

    img = Image.open(BytesIO(decoded))
    if img.mode != 'L':
        img = img.convert('L')

    img_resized = img.resize(IMAGE_SIZE, resample=Image.LANCZOS)
    img_resized = np.invert(img_resized) / 255  # bitwise
    img_array = np.array(img_resized).reshape(1, 28, 28, 1).astype('float64')

    return np.array_str(np.argmax(model.predict(img_array), axis=1))


if __name__ == "__main__":
    app.run(debug=True)
