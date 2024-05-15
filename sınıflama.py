import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
import PIL
import PIL.Image

# dosya yolu
toplam_res = 0
dosya_yolu ="model"
dirs = os.listdir(dosya_yolu)
for dir in dirs:
     dosya = list(os.listdir('model/'+dir))


     #toplam_res = toplam_res + len(dosya)
     # dosya kontrol için
     #print(dir + ' dosyasında  ' + str(len(dosya)) + ' resim ')
     #print(str(toplam_res) + ' resim
"""cicekler = [os.path.join(dosya_yolu, "Lale", f) for f in os.listdir(os.path.join(dosya_yolu, "Lale"))]
img = PIL.Image.open(cicekler[73])
img.show()"""


batch_size = 32
img_height = 180
img_width = 180
# TensorFlow ile veri kümesini yükleme
train_ds = tf.keras.utils.image_dataset_from_directory(dosya_yolu,
                                                       validation_split=0.2,
                                                       subset="training",
                                                       seed=123,
                                                       image_size=(img_height,img_width),
                                                       batch_size=batch_size)

