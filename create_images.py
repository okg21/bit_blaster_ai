import cv2 as cv
import numpy as np

data = np.load("D:/bb_data/training_data.npy", allow_pickle=True)
targets = np.load("D:/bb_data/target_data.npy", allow_pickle=True)

#print shapes of data
print(f'Image Data Shape: {data.shape}')
print(f'targets Shape: {targets.shape}')


# Lets see how many of each type of move we have.
unique_elements, counts = np.unique(targets, return_counts=True)
print(np.asarray((unique_elements, counts)))

holder_list = []
for i, image in enumerate(data):
    holder_list.append([data[i], targets[i]])

count_up = 0
count_left = 0
count_right = 0
count_down = 0

for data in holder_list:
    if data[1] == 'W':
        count_up += 1
        cv.imwrite(f"D:/BB/Up/H7-u{count_up}.png", data[0]) 
    elif data[1] == 'A':
        count_left += 1
        cv.imwrite(f"D:/BB/Left/H7-l{count_left}.png", data[0]) 
    elif data[1] == 'D':
        count_right += 1
        cv.imwrite(f"D:/BB/Right/H7-r{count_right}.png", data[0]) 
    elif data[1] == 'S':
        count_down += 1
        cv.imwrite(f"D:/BB/Down/H7-d{count_down}.png", data[0]) 