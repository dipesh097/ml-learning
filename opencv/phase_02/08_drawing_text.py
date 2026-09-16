
import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

cv2.putText(image,"hello i am dipesh",(70,90),cv2.FONT_HERSHEY_COMPLEX,2.0,(256,200,256),10)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()