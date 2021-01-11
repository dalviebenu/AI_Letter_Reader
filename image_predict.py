import cv2
from model import *
from pathlib import Path
import scipy as sp
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image, ImageChops
import os


def filter_img(img):
    # Img is an array, mimics the filter done on the training data
    return sp.ndimage.gaussian_filter(img, 1)


def img_as_array():
    # Loads img, inverts and converts it to an array. Assumes image is black on white
    img = load_img('./src/temp/temp.png', grayscale=True)
    img = np.invert(img)
    img = img_to_array(img)
    return img


def make_square(file):
    # takes an image and makes it square, returns an array
    img = load_img(file, grayscale=True)
    height = img.size[1]
    width = img.size[0]
    base = height if height >= width else width
    background = Image.new('RGBA', (base + 30, base + 30), (255, 255, 255, 255))
    offset = (round(((base + 30) - img.size[0]) / 2), round(((base + 30) - img.size[1]) / 2))
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
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    img = img.reshape(1, 28, 28, 1)
    return img


def test_images():
    file = './src/letters/1_1_3.png'
    model = load_model('final_model_letters.h5')
    trim(file)
    test = pre_process_img(file)

    # A = create_image('A.png')
    print(test.shape)
    out = model.predict_classes(test)
    print(alpha[out[0]])


def path_of_imgs(path='./src/letters'):
    pathlist = Path(path)
    paths = []

    stems = [file.stem for file in pathlist.iterdir()]
    del stems[0]  # .donotdelete
    stems.sort(key=int)

    for stem in stems:
        paths.append(path + '/' + stem + '.png')

    return paths


def trim(file):
    im = Image.open(file)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        temp = im.crop(bbox)
        temp.save(file)
        return im.crop(bbox)


def test_list_images(model='final_model_5.h5', img_dir_path='./src/letters'):
    # paths = Path(img_dir_path).glob('*.png')
    paths = path_of_imgs()
    model_inst = load_model(model)
    image_arr_list = []
    list_of_out = []

    for file in paths:
        trim(file)
        temp = pre_process_img(file)
        image_arr_list.append(temp)

    for test in image_arr_list:
        out = model_inst.predict_classes(test)
        list_of_out.append(OUTPUT[out[0]])

    print(list_of_out)

