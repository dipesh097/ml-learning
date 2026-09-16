
import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")
# -----------------------------rectangle drawoing---------------------------------
cv2.rectangle(image,(200,300),(600,900),(256,255,254))
cv2.imshow("rectangular",image)
# ------------------------------------circle drawing-----------------------------------

cv2.circle(image,(200,200),150,(0,255,0),55)

cv2.imshow("circle",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

