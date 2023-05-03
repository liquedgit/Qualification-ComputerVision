import cv2
import os
import random
from ImageProcess import *
from ShapeDetect import shape_detect
from EdgeDetect import *
from train import *
# from FaceDetect import livecam

imagelist = []
for index, img in enumerate(os.listdir('./assets')):
    path = f'./assets/{img}'
    # print(path)
    imagelist.append(path)


def menuChoice():
    inputuser = 0
    while(inputuser != 5):
        print('1. Process Random image to be grayscale')
        print('2. Threshold random image with otsu algorithm')
        print('3. Threshold random image with binary algorithm')
        print('4. Blur random image using median blur')
        print('5. back to menu')
        inputuser = input('> ')
        if not inputuser.isdigit():
            continue
        inputuser = int(inputuser)
        index_random = random.randint(0, len(imagelist)-1)
        randimg = cv2.imread(imagelist[index_random])
        if inputuser == 1:
            cv2.imshow('GrayScale Filter', grayscale_filter(randimg))
        elif inputuser == 2:
            cv2.imshow('Otsu Threshold', threshold_otsu(randimg))
        elif inputuser == 3:
            cv2.imshow('Binary Threshold', threshold_binary(randimg))
        elif inputuser == 4:
            cv2.imshow('Median Blur Filter', blur_filter(randimg))
        cv2.waitKey(0);
        cv2.destroyAllWindows()
    return


print('Training model...')
train_model()

choice = 0
while choice != 5:
    print('Select menu :')
    print('1. Image Process')
    print('2. Edge Detection')
    print('3. Shape Detection')
    print('4. Show live camera and Recognition the faces')
    print('5. Exit')
    
    choice = input('> ')
    if not choice.isdigit():
        continue
    choice = int(choice)
    if choice == 1:
        menuChoice()
    elif choice ==2:
        rand = random.randint(0, len(imagelist)-1)
        currimg = cv2.imread(imagelist[rand])
        detectEdge(currimg)
    elif choice ==3:
        rand = random.randint(2,3)
        shapeimg = cv2.imread(imagelist[rand])
        cv2.imshow('Shape recognition',shape_detect(shapeimg))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif choice == 4:
        livecam()
        
    



# img = cv2.imread('assets/lena.jpg')
# shape = cv2.imread('./assets/shapes.png')
# shapeDetect = shape_detect(shape)

# cv2.imshow('Shape detected', shapeDetect)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Blur', blur_filter(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('threshold', threshold_binary(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
