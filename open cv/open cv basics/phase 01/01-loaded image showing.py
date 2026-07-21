import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20260409_01_27_28_Pro.jpg")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow("hello",gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
