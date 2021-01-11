from model import *
from image_predict import *
from segmentation import *


def run():
    segmentation(image='./src/test4.png')
    test_list_images()


if __name__ == '__main__':
    run()
