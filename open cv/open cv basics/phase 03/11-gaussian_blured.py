import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg")

blur_image=cv2.GaussianBlur(image,(9,9),10)

cv2.imshow("original image",image)
cv2.imshow("blured image",blur_image)

cv2.waitKey(0) 

cv2.destroyAllWindows()
    

