import sys
import os
import glob
import cv2
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def show_images(src_img, bw_img, img_f):
    cv2.namedWindow('Src', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Src", src_img)
    cv2.namedWindow('Converted to BW', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Converted to BW", bw_img)
    cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Threshold", img_f)


def close_images():
    k = cv2.waitKey(0)
    if k & 0xFF == ord('q'):
        print("done")
    elif k & 0xFF == int(27):
        cv2.destroyAllWindows()
    else:
        close_images()


def line_array(array):
    """Utility function to find the region to draw the horizontal lines

    @array - input array\n
    @Returns int array containing approximate y coordinates of the lines\n
    """
    x_upper = []
    x_lower = []
    for i in range(1, len(array) - 1):
        start_a, start_b = draw_line(i, array)
        end_a, end_b = stop_line(i, array)
        if start_a >= 7 and start_b >= 5:
            x_upper.append(i)
        if end_a >= 5 and end_b >= 7:
            x_lower.append(i)
    return x_upper, x_lower


def draw_line(y, array):
    """Utility function to find the beginning and the end of the white pixels on the image, use to find beginning of the line\n

    @y coord index\n
    @array array with the image data\n
    @Returns beginning and the end coordinates of the beginning and the end\n
    """
    next = 0
    prev = 0
    for val in array[y: (y + 10)]:
        if val > 3:
            next += 1
    for val in array[(y - 10):y]:
        if val == 0:
            prev += 1
    return next, prev


def stop_line(y, array):
    """Utility function to find the beginning and the end of the white pixels on the image, use to find end of the line\n

    @y coord index\n
    @array array with the image data\n
    @Returns beginning and the end coordinates of the beginning and the end\n
    """
    next = 0
    prev = 0
    for i in array[y:y + 10]:
        if i == 0:
            next += 1
    for i in array[y - 10:y]:
        if i > 3:
            prev += 1
    return next, prev


def endline_word(y, array, width):
    """Utility function to find the beginning and the end of the white pixels on the image, use to find end of the line\n

    @array - array with the image data\n
    @width - average letter width\n
    @Returns array of the end lines\n
    """
    next = 0
    prev = 0
    for i in array[y:y + 2 * width]:
        if i < 2:
            next += 1
    for i in array[y - width:y]:
        if i > 2:
            prev += 1
    return prev, next


def end_line_array(array, width):
    """Utility function to find the beginning and the end of the white pixels on the image, use to find end of the line\n

    @array - array with the image data\n
    @width - average letter width\n
    @Returns array of the end lines\n
    """
    list = []
    for y in range(len(array)):
        e_p, e_a = endline_word(y, array, width)
        if e_a >= int(1.5 * width) and e_p >= int(0.7 * width):
            list.append(y)
    return list


def shrink_endline(array):
    """Utility function to find exact location of vertical line from a region\n

    @array - array vertical line bound\n
    @Returns exact coordinate of vertical line\n
    """
    list = []
    for i in range(len(array) - 1):
        if array[i] + 1 < array[i + 1]:
            list.append(array[i])
    list.append(array[-1])
    return list


def shrink_array(u, l):
    """Utility function provide exact location of a horizontal lines\n

    @u upper lines region\n
    @l lower lines region\n
    @Returns arrays with y coordinates of horizontal lines\n
    """
    upperlines = []
    lowerlines = []
    for i in range(len(u) - 1):
        if u[i] + 3 < u[i + 1]:
            upperlines.append(u[i] - 5)

    for i in range(len(l) - 1):
        if l[i] + 3 < l[i + 1]:
            lowerlines.append(l[i] + 5)
    upperlines.append(u[-1] - 3)
    lowerlines.append(l[-1] + 3)
    return upperlines, lowerlines


def e_width(contours):
    """Utility function to count average width of an element on an image\n

    @Args array from cv2 contours\n
    @Returns mean letter width and standard deviation of the element\n
    """
    sum = []
    i = 0
    for n in contours:
        # ignore small characters due to noise
        if cv2.contourArea(n) > 250:
            x, y, w, h = cv2.boundingRect(n)
            sum.append(w)
            i += 1

    return np.mean(sum), np.std(sum)


def word_end_detect(lines, i, img, width, img_f, img_w):
    """Utility function for segmenting words\n

    @Args:
     lines - horizontal lines coords \n
     i - Counter element \n
     img - target image
     m_w - letter medium width
    @Returns coordinates of vertical lines\n
    """
    # search vertically for pixels, if encounter one then draw a line
    search_y = np.zeros(shape=img_w)
    for x in range(img_w):
        for y in range(lines[i][0], lines[i][1]):
            if img[y][x] == 255:
                search_y[x] += 1
    l = end_line_array(search_y, int(width))

    endL = shrink_endline(l)
    # draw a vertical line
    for x in endL:
        img_f[lines[i][0]:lines[i][1], x] = 255
    return endL


def letter_seg(img, x_lines, i, lines):
    """Utility function for segmenting letters\n

    @Args:
     Image line array \n
     N-dimensional array with coords of the vertical lines on each horizontal line N \n
     Counter element \n
    @Returns nothing\n
    """
    img_c = img[i].copy()
    x_l_c = x_lines[i].copy()

    letters = []

    contours, hierarchy = cv2.findContours(img_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    for j in contours:
        if cv2.contourArea(j) > 200:
            x, y, w, h = cv2.boundingRect(j)
            h = lines[i][1] - lines[i][0] - 2
            letters.append((x, 5, w, h))

    # sort tuples based on their x coordinate
    letters = sorted(letters, key=lambda student: student[0])
    word = 1
    letter_i = 0
    # get the proper index for each letter in format line_word_letter
    for k in range(len(letters)):
        if letters[k][0] < x_l_c[0]:
            letter_i += 1
        else:
            x_l_c.pop(0)
            word += 1
            letter_i = 1

        # extract letter from the main picture
        letter_pic = img[i][letters[k][1] - 5:letters[k][1] + letters[k][3] + 5,
                     letters[k][0] - 5:letters[k][0] + letters[k][2] + 5]
        # save it
        cv2.imwrite('./src/letters/' + str(i + 1) + '_' + str(word) + '_' + str(letter_i) + '.png',
                    255 - letter_pic)


def segmentation():

    print("Start!")
    # clean the directory
    files = glob.glob('./src/letters/*.png')
    for f in files:
        os.remove(f)
    # specify the file location
    src_img = cv2.imread('./src/test4.png')
    height, width = src_img.shape[0], src_img.shape[1]

    # resize the image
    src_img = cv2.resize(src_img, dsize=(1280, int(1280 * height / width)), interpolation=cv2.INTER_AREA)
    height, width = src_img.shape[0], src_img.shape[1]
    print("Height = ", height, ", Width = ", width)

    # convert to greyscale and apply de-noising
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    cv2.fastNlMeansDenoising(grey_img, grey_img, 20)

    # use thresholding to convert image to black and white
    bw_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 41, 39)

    # define kernel type and size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # apply more targeted de-noising
    img_f = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
    img = img_f.copy()

    # find white pixels in the image to segment text on the image into the lines
    find_x = np.zeros(shape=(height))
    for i in range(height):
        for j in range(width):
            if bw_img[i][j] == 255:
                find_x[i] = find_x[i] + 1

    # get the y positions of the lines
    up_l, bottom_l = line_array(find_x)
    upL, bottomL = shrink_array(up_l, bottom_l)

    if len(upL) == len(bottomL):
        lines = []
        # draw lines
        for i in upL:
            img_f[i][:] = 255
        for i in bottomL:
            img_f[i][:] = 255
        for i in range(len(upL)):
            lines.append((upL[i], bottomL[i]))
    lines = np.array(lines)
    # image with lines
    l_img = []
    for i in range(len(lines)):
        l_img.append(bw_img[lines[i][0]:lines[i][1], :])

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    cv2.drawContours(src_img, contours, -1, (40, 106, 237), 1)

    # Do word detection and segmentation on each line
    mean, std = e_width(contours)
    print(mean, std)
    x_lines = []

    for i in range(len(l_img)):
        x_lines.append(word_end_detect(lines, i, bw_img, mean * 0.75, img_f, int(width)))
    for i in range(len(x_lines)):
        x_lines[i].append(width)

    # segment letters on each line and save their images
    for i in range(len(lines)):
        letter_seg(l_img, x_lines, i, lines)

    show_images(src_img, bw_img, img_f)
    close_images()


segmentation()
