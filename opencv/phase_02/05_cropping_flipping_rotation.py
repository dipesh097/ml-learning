
import cv2

image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

if image is None:
    print("image is not loaded")

else :
        #  --------------------croping---------------------------------
    print("image loaded successfully")
    
    # cropped=image[50:500,50:5000]
    # cv2.imshow("original",image)

    # cv2.imshow("cropped",cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#  rotaion

h,w=image.shape[:2]

centre=(h//2,w//2)

M=cv2.getRotationMatrix2D(centre,70,0.5)
roteded=cv2.warpAffine(image,M,(w,h))
cv2.imshow("roteded",roteded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------flipin----------------------------------------------------

verticle_fliping=cv2.flip(image,0)
horizontal_fliping=cv2.flip(image,1)
both_fliping=cv2.flip(image,-1)
cv2.imshow("verticle_fliping",verticle_fliping)
cv2.imshow("horizontal_fliping",horizontal_fliping)
cv2.imshow("both_fliping",both_fliping)
cv2.waitKey(0)
cv2.destroyAllWindows()


