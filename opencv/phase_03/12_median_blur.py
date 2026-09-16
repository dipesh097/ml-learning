import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg")

blured=cv2.medianBlur(image,13)

cv2.imshow("original image",image)
cv2.imshow("blured image",blured)

cv2.waitKey(0) 

cv2.destroyAllWindows()
    