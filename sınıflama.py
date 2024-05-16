import cv2
from PIL import Image
import os



data_dir = 'model'
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


