import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

gray_image=cv2.cvtColor("gray-image",cv2.bgr2gray)

cv2.imshow("hello",image)
cv2.waitKey(0)
cv2.destryAllWindows()
