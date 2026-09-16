
import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

cv2.line(image,(100,500),(1000,800),(0,255,255),20)
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()