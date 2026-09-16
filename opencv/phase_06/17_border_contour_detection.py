import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg")

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

_,threshold=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image,contours,-1,255,3)

cv2.imshow("with contours",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
