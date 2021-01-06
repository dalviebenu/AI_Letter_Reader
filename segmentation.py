import cv2
import numpy as np

def show_images():
    cv2.namedWindow('Src', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Src", src_img)
    cv2.namedWindow('Converted to BW', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Converted to BW", bw_img)
    cv2.namedWindow('Thresholded', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Thresholded", img_f)


def close_images():
    k = cv2.waitKey(0)
    if k & 0xFF == ord('s'):
        print("Completed")
    elif k & 0xFF == int(27):
        cv2.destroyAllWindows()
    else:
        close_images()


def line_array(array):
    list_x_upper = []
    list_x_lower = []
    for y in range(5, len(array) - 5):
        s_a, s_p = strtline(y, array)
        e_a, e_p = endline(y, array)
        if s_a >= 7 and s_p >= 5:
            list_x_upper.append(y)
        if e_a >= 5 and e_p >= 7:
            list_x_lower.append(y)
    return list_x_upper, list_x_lower


def strtline(y, array):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y + 10]:
        if i > 3:
            count_ahead += 1
    for i in array[y - 10:y]:
        if i == 0:
            count_prev += 1
    return count_ahead, count_prev


def endline(y, array):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y + 10]:
        if i == 0:
            count_ahead += 1
    for i in array[y - 10:y]:
        if i > 3:
            count_prev += 1
    return count_ahead, count_prev


def endline_word(y, array, a):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y + 2 * a]:
        if i < 2:
            count_ahead += 1
    for i in array[y - a:y]:
        if i > 2:
            count_prev += 1
    return count_prev, count_ahead


def end_line_array(array, a):
    list_endlines = []
    for y in range(len(array)):
        e_p, e_a = endline_word(y, array, a)
        # print(e_p, e_a)
        if e_a >= int(1.5 * a) and e_p >= int(0.7 * a):
            list_endlines.append(y)
    return list_endlines


def refine_endword(array):
    refine_list = []
    for y in range(len(array) - 1):
        if array[y] + 1 < array[y + 1]:
            refine_list.append(array[y])
    refine_list.append(array[-1])
    return refine_list


def refine_array(array_upper, array_lower):
    upperlines = []
    lowerlines = []
    for i in range(len(array_upper) - 1):
        if array_upper[i] + 5 < array_upper[i + 1]:
            upperlines.append(array_upper[i] - 10)
    for i in range(len(array_lower) - 1):
        if array_lower[i] + 5 < array_lower[i + 1]:
            lowerlines.append(array_lower[i] + 10)

    upperlines.append(array_upper[-1] - 10)
    lowerlines.append(array_lower[-1] + 10)

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


def end_wrd_dtct(lines, i, img, m_w):
    """Utility function for segmenting words\n

    @Args:
     Image line array \n
     Counter element \n
     N-dimensional array with coords of the vertical lines on each horizontal line N \n
    @Returns nothing\n
    """
    #search vertically for pixels, if encounter one then draw a line
    search_y = np.zeros(shape=width)
    for x in range(width):
        for y in range(lines[i][0], lines[i][1]):
            #basically going vertically between the horizontal lines
            if img[y][x] == 255:
                search_y[x] += 1
    l = end_line_array(search_y, int(m_w))

    endL = refine_endword(l)
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
        if cv2.contourArea(j) > 250:
            x, y, w, h = cv2.boundingRect(j)
            h = lines[i][1]-lines[i][0]-2
            print(y)
            letters.append((x, 5, w, h))




    # sort tuples based on their x coordinate
    letters = sorted(letters, key=lambda student: student[0])
    print(letters)
    word = 1
    letter_i = 0
    #get the proper index for each letter in format line_word_letter
    for k in range(len(letters)):
        if letters[k][0] < x_l_c[0]:
            letter_i += 1
        else:
            x_l_c.pop(0)
            word += 1
            letter_i = 1

        #extract letter from the main picture
        letter_pic = img[i][letters[k][1] - 5:letters[k][1] + letters[k][3] + 5, letters[k][0] - 5:letters[k][0] + letters[k][2] + 5]
        #save it
        cv2.imwrite('./src/letters/' + str(i + 1) + '_' + str(word) + '_' + str(letter_i) + '.png',
                    255 - letter_pic)


print("Start!")
src_img = cv2.imread('./src/test.png', 1)
height, width = src_img.shape[0], src_img.shape[1]

# resize the image
src_img = cv2.resize(src_img, dsize=(1280, int(1280 * height/width)), interpolation=cv2.INTER_AREA)
height, width = src_img.shape[0], src_img.shape[1]
print("Height = ", height, ", Width = ", width)

# convert to greyscale
grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
# use thresholding to convert image to black and white
bw_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 20)
bw_img1 = bw_img.copy()
bw_img2 = bw_img.copy()
# define structure element of a character as an ellipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
# and denoise image
img_f = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
img = img_f.copy()


# find white pixels in the image to segment text on the image into the lines
find_x = np.zeros(shape=(height))
for y in range(height):
    for x in range(width):
        if bw_img[y][x] == 255:
            find_x[y] = find_x[y] + 1

up_l, bottom_l = line_array(find_x)

upL, bottomL = refine_array(up_l, bottom_l)

if len(upL) == len(bottomL):
    lines = []
    for y in upL:
        img_f[y][:] = 255
    for y in bottomL:
        img_f[y][:] = 255
    for y in range(len(upL)):
        lines.append((upL[y], bottomL[y]))
lines = np.array(lines)
# image with lines
l_img = []
for i in range(len(lines)):
    l_img.append(bw_img2[lines[i][0]:lines[i][1], :])

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

final_contr = np.zeros((img_f.shape[0], img_f.shape[1], 3), dtype=np.uint8)
cv2.drawContours(src_img, contours, -1, (40, 106, 237), 1)

# Do word detection and segmentation on each line
mean, std = e_width(contours)
print(mean, std)
x_lines = []
for i in range(len(l_img)):
    x_lines.append(end_wrd_dtct(lines, i, bw_img, mean*0.75))

for i in range(len(x_lines)):
    x_lines[i].append(width)
#on each line
for i in range(len(lines)):
    letter_seg(l_img, x_lines, i, lines)

show_images()
close_images()
