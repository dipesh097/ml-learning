import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg",cv2.IMREAD_GRAYSCALE)

ret , thersold=cv2.threshold(image,100,255,cv2.THRESH_BINARY)

cv2.imshow("original image",image)
cv2.imshow("edge image",thersold)
cv2.imwrite("output of 15.png",thersold)



cv2.waitKey(0) 

cv2.destroyAllWindows()