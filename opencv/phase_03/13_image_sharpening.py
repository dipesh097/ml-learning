import cv2

import numpy as np

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg")

kernel=np.array([
              [0,-1,0],
              [-1,5,-1],
              [0,-1,0]
                ]    
                )

sharpned=cv2.filter2D(image,-5,kernel)


cv2.imshow("original image",image)
cv2.imshow("shaprpend image",sharpned)



cv2.waitKey(0) 

cv2.destroyAllWindows()
    