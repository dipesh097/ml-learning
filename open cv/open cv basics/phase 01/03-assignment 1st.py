import cv2
image=cv2.imread(r"C:\Users\dipesh's-device\Pictures\Camera Roll\Screenshots\Screenshot 2025-10-27 210150.png")

if image is None:
    print("loaded SUCCESSfully")
    exit()

h,w,c=image.shape

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

print("""hello user \n what you want \n 1 : save image \n 2 :show gray image""")

user=int(input("here(gives reply in terms of number):"))
 

if user==1:
    print("enter adrees where you want save it (note:ending must be .png,.spg etc)")
    adress=(input("enter adress :"))
    cv2.imwrite(adress,gray_image)

elif user==2:
     cv2.imshow("hello",gray_image)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
else :
    print("occurs error")