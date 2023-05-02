import cv2


def grayscale_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# def


def blur_filter(img):
    return cv2.medianBlur(img, 13)
