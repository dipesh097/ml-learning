#   height and width of img 1 and img 2  should be  same
import cv2
import numpy as np

img1=np.zeros((300,300),dtype="uint8")
img2=np.zeros((300,300),dtype="uint8")

circle=cv2.circle(img1,(150,150),100,255,-1)
rectangle=cv2.rectangle(img2,(100,100),(250,250),255,-1)

cv2.imshow("circle",circle)
cv2.imshow("rectangle",rectangle)

cv2.imshow("rectangle and circle with and",cv2.bitwise_and(img1,img2))

cv2.imshow("rectangle and circle with or",cv2.bitwise_or(img1,img2))

cv2.imshow("rectangle with not",cv2.bitwise_not(img2))


cv2.waitKey(0)
cv2.destroyAllWindows()

