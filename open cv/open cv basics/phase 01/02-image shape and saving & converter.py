import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

h,w,c=image.shape

print(fr"height-{h},width-{w},channel-{c}")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray-image.png",gray_image)

cv2.imshow("gray-image",gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
