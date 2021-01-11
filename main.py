from model import *
from image_predict import *
from segmentation import *


def run():
    segmentation(image='./src/test9.png')
    test_list_images(model='final_model_5.h5')


if __name__ == '__main__':
    run()