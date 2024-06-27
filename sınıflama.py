import cv2
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#veriyi okuma
data_dir = 'data'
data_list = os.listdir(os.path.join(data_dir))
print(data_list)
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
            x = 1
            #print('Issue with image {}'.format(image_path))
# veriyi yükleme
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# batch de ki veriyi 0 ve 1 arasına normalize etme
batch = batch[0] / 255.0
print(batch[0].max())
print(batch[0].min())
# veriyi eğitim ve doğrulama olarak ayırma
train_data = tf.keras.utils.image_dataset_from_directory('data',
                                                         validation_split=0.2,
                                                         subset='training',
                                                         seed=123)
val_data = tf.keras.utils.image_dataset_from_directory('data',
                                                         validation_split=0.2,
                                                         subset='validation',
                                                         seed=123)
class_names = train_data.class_names
print(class_names)
# veri setinden ilk 9 resmi gösterme
"""plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        plt.show()"""
#performans için veri setini optimize etme
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

# modeli oluşturma

num_classes = len(class_names)
input_shape = (256, 256, 3)
model = Sequential([
  layers.Rescaling(1./255, input_shape=input_shape),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
# modeli derleme
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())
# modeli eğitme
epochs=10
history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs
)
#eğitim sonucunu görselleştirme
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(8,8))
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label=' Eğitim doğruluğu (Training Accuracy)')
plt.plot(epochs_range, val_acc, label='Doğrulama doğruluğu (Validation Accuracy)')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim kaybı (Training Loss)')
plt.plot(epochs_range, val_loss, label='Doğrulama kaybı (Validation Loss)')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
