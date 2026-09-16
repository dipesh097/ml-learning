import cv2

import numpy as np

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg",cv2.IMREAD_GRAYSCALE)

edge=cv2.Canny(image,60,180)

cv2.imshow("original image",image)
cv2.imshow("edge image",edge)
cv2.imwrite("output.png",edge)



cv2.waitKey(0) 

cv2.destroyAllWindows()
    