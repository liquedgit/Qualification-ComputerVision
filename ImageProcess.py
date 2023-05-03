import cv2


def grayscale_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold_otsu(img):
    gray = grayscale_filter(img)
    _, thresh_img = cv2.threshold(gray, 0 , 255, cv2.THRESH_OTSU)
    return thresh_img

def threshold_binary(img):
    gray = grayscale_filter(img)
    _, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY);
    return thresh_img

def blur_filter(img):
    return cv2.medianBlur(img, 13)

def spesific_image_blur(img, x, y, w, h):
    return cv2.medianBlur(img[y:y+w, x:x+h], 21)


