import tensorflow as tf
import numpy as np
import tensorflow.lite as tflite

from tensorflow import keras
from io import BytesIO
from urllib import request
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input


#get_ipython().system('wget https://github.com/alexeygrigorev/large-datasets/releases/download/wasps-bees/bees-wasps.h5 -O homework-model.h5')

model = keras.models.load_model('homework-model.h5', compile=False)

#get_ipython().system('pip install pillow')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

def preprocessing(url):
    img = download_image(url)
    target_size = (150, 150)

    img = prepare_image(img, target_size)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    datagen = ImageDataGenerator(rescale=1./255)
    preprocessed_img = datagen.flow(img_array)

    for batch in preprocessed_img:
        preprocessed_img = batch
        break
    return preprocessed_img

#preds = model.predict(preprocessed_img)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('homework.tflite', 'wb') as f_out:
  f_out.write(tflite_model)

interpreter = tflite.Interpreter(model_path='homework.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):
    X = preprocessing(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return preds