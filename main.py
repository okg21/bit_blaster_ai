import cv2 as cv
import numpy as np
import os
import time
from windowcapture import WindowCapture
from vision import Vision
from getkeys import key_check

os.chdir(os.path.dirname(os.path.abspath(__file__)))
file_name = "D:/bb_data/training_data.npy"
file_name2 = "D:/bb_data/target_data.npy"

# initialize the WindowCapture class
wincap = WindowCapture('Bit Blaster XL')
# initialize the Vision class
vision_score = Vision('score.jpg')

def get_data():

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        image_data = list(np.load(file_name, allow_pickle=True))
        targets = list(np.load(file_name2, allow_pickle=True))
    else:
        print('File does not exist, starting fresh!')
        image_data = []
        targets = []
    return image_data, targets


def save_data(image_data, targets):
    np.save(file_name, image_data)
    np.save(file_name2, targets)


image_data, targets = get_data()
while True:
    keys = key_check()
    print("waiting press B to start")
    if keys == "B":
        print("Starting")
        break


count = 0
loop_time = time.time()
while(True):
    count += 1
    last_time = time.time()
    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    # convert the image to grayscale
    gray_screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    # apply canny edge detection
    canny_screenshot = cv.Canny(gray_screenshot, threshold1=119, threshold2=250)
    # resize
    resized_screenshot = cv.resize(canny_screenshot, (224,224))

    cv.imshow("canny", resized_screenshot)
    cv.waitKey(1)

    #convert to numpy array
    image = np.array(resized_screenshot)
    image_data.append(image)

    keys = key_check()
    targets.append(keys)
    print(keys)
    if keys == "H":
        break
    print('loop time : {} '.format(time.time()-last_time))

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
save_data(image_data, targets)
print('Done.')