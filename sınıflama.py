import cv2
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image

#veriyi okuma
data_dir = 'data'
data_dir1 = 'C:/Users/ASUSS/Desktop/proje/cıcek_sınıflama/data'
data_list = os.listdir(os.path.join(data_dir))
#print(data_list)
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in ['jpeg', 'jpg', 'png']:
                print('resim ext liste yok {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
# veriyi yükleme
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = model.as_numpy_iterator()
batch = data_iterator.next()
print(len(batch))