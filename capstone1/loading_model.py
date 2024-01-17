
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[2]:


model = keras.models.load_model('xception_v4_1_23_0.683.h5', compile=False)


# In[ ]:


path = './dogs/images/n02085936-Maltese_dog/n02085936_37.jpg'
img = load_img(path, target_size=(150, 150))


# In[20]:


x = np.array(img)
X = np.array([x])

X = preprocess_input(X)


# In[8]:


pred = model.predict(X)


# In[10]:


class_names = sorted(os.listdir("./dogs/images"))


# In[11]:


result = list(zip(class_names, pred[0]))
for i in range(len(result)):
    if (result[i][1] >= 1):
        print(result[i])


# In[12]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('dog-classification-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[13]:


import tensorflow.lite as tflite


# In[14]:


interpreter = tflite.Interpreter(model_path='dog-classification-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[21]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
pred = interpreter.get_tensor(output_index)


# In[22]:


result = list(zip(class_names, pred[0]))
for i in range(len(result)):
    if (result[i][1] >= 1):
        print(result[i])


# In[8]:


import tflite_runtime.interpreter as tflite
import os

from PIL import Image
from keras_image_helper import create_preprocessor


# In[9]:


class_names = sorted(os.listdir("./dogs/images"))


# In[10]:


interpreter = tflite.Interpreter(model_path='dog-classification-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[26]:


preprocessor = create_preprocessor('xception', target_size=(150, 150))

url = 'http://3.bp.blogspot.com/-ZXixpl4wbCE/UIXNI2jDBfI/AAAAAAAABDY/H8l5InQBgHI/s1600/Rottweiler-Puppy-Picture.JPG'
X = preprocessor.from_url(url)


# In[27]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
pred = interpreter.get_tensor(output_index)


# In[28]:


result = list(zip(class_names, pred[0]))
for i in range(len(result)):
    if (result[i][1] >= 1):
        print(result[i])


# In[ ]:




