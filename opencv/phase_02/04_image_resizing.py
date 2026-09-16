
import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

if image  is None:
    print("image is not loaded")
else:
    print("image is successfully loaded")
    
    resized=cv2.resize(image,(400,400))
    cv2.imshow("oringinal image",image)
    cv2.imshow("resized image",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output.png",resize)