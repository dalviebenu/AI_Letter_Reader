import cv2
from model import *
from pathlib import Path
import scipy as sp
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image, ImageFilter


def filter_img(img):
    # Img is an array, mimics the filter done on the training data
    return sp.ndimage.gaussian_filter(img, 1)


def img_as_array():
    # Loads img, inverts and converts it to an array. Assumes image is black on white
    img = load_img('test.png', grayscale=True)
    img = np.invert(img)
    img = img_to_array(img)
    return img


def make_square(file):
    # takes an image and makes it square, returns an array
    img = load_img(file, grayscale=True)
    height = img.size[1]
    width = img.size[0]
    base = height if height >= width else width
    background = Image.new('RGBA', (base, base + 22), (255, 255, 255, 255))
    offset = (round((base - img.size[0]) / 2), round(((base + 22) - img.size[1]) / 2))
    background.paste(img, offset)
    background.save('./src/temp/temp.png')  # temp image


def down_sample(img):
    # Takes an image array and down samples it to a 28 X 28 image
    return cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)


def pre_process_img(file):
    make_square(file)
    img = img_as_array()
    img = filter_img(img)
    img = down_sample(img)

    img = img.astype('float32')
    img = img / 255.0
    img = img.reshape(28, 28, 1)
    # plt.imshow(img, cmap=plt.get_cmap('gray'))
    # plt.show()
    img = img.reshape(1, 28, 28, 1)
    return img


def test_images():
    model = load_model('final_model_4.h5')
    test = pre_process_img('./src/letters/1_1_1.png')

    # A = create_image('A.png')
    print(test.shape)
    out = model.predict_classes(test)
    print(OUTPUT[out[0]])


def test_list_images():
    pathlist = Path('./src/letters').rglob('*.png')
    model = load_model('final_model_5.h5')
    image_arr_list = []
    list_of_out = []

    for path in pathlist:
        temp = pre_process_img(path)
        image_arr_list.append(temp)

    for test in image_arr_list:
        out = model.predict_classes(test)
        list_of_out.append(OUTPUT[out[0]])

    print(list_of_out)


test_list_images()
