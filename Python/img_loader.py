import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

correct_images_path = os.path.join(os.path.curdir, 'imgs_correct')
incorrect_images_path = os.path.join(os.path.curdir, 'imgs_incorrect')


def load_correct_images():
    result = []
    for img_name in get_file_list(correct_images_path):
        img_path = os.path.join(correct_images_path, img_name)
        img = Image.open(img_path).convert('L')
        im2arr = convert_image(img)
        result.append({'raw': img, 'img': im2arr, 'label': img_name[:1]})

    return result


def load_incorrect_images():
    result = []
    for img_name in get_file_list(incorrect_images_path):
        img_path = os.path.join(incorrect_images_path, img_name)
        img = Image.open(img_path).convert('L')
        im2arr = convert_image(img)
        result.append({'raw': img, 'img': im2arr, 'label': img_name[:-4]})

    return result


def convert_image(image):
    resized_img = np.resize(image, (28, 28, 1))
    im2arr = np.array(resized_img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    im2arr = (255 - im2arr.astype('float32')) / 255
    return im2arr


def get_file_list(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f[-4:] == '.png']

