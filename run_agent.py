
import random
import time

from getkeys import key_check
import pydirectinput
import keyboard
import time
import cv2
from windowcapture import WindowCapture
from direct_keys import PressKey, ReleaseKey, W, D, A
from fastai.vision.all import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def label_func(x): return x.parent.name
learn_inf = load_learner("C:/Users/omerk/Desktop/reinforcement learning/bit_blaster/export.pkl")
print("loaded learner")

# initialize the WindowCapture class
wincap = WindowCapture('Bit Blaster XL')

sleepy = 0.1
# Wait for me to push B to start.
keyboard.wait('B')
time.sleep(sleepy)


while True:

    image = wincap.get_screenshot()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, threshold1=200, threshold2=300)
    image = cv2.resize(image,(224,224))
    # cv2.imshow("Fall", image)
    # cv2.waitKey(1)
    start_time = time.time()
    result = learn_inf.predict(image)
    action = result[0]
    #print(result[2][0].item(), result[2][1].item(), result[2][2].item(), result[2][3].item())
    print(action)
    
    if action == "Up" or result[2][0]>.1:
        print(f"Up! - {result[1]}")
        keyboard.press("w")
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("s")
        time.sleep(sleepy)

    if action == "Down":
        print(f"DOWN! - {result[1]}")
        keyboard.press("s")
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("w")
        time.sleep(sleepy)

    elif action == "Left":
        print(f"LEFT! - {result[1]}")
        keyboard.press("a")
        keyboard.release("d")
        keyboard.release("w")
        keyboard.release("s")
        time.sleep(sleepy)

    elif action == "Right":
        #print(f"Right! - {result[1]}")
        keyboard.press("d")
        keyboard.release("a")
        keyboard.release("w")
        keyboard.release("s")
        time.sleep(sleepy)


    # End simulation by hitting h
    keys = key_check()
    if keys == "H":
        break

#keyboard.release('W')
