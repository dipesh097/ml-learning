"""Build:

👉 Webcam Edge Detection App

Features:

Open webcam using OpenCV

Apply Canny edge detection in real-time

Add keyboard controls:

press e → edge ON/OFF

press q → quit"""



import cv2
import time

cap=cv2.VideoCapture(0)

edge_on=False
previous_time=0  #for calculatin time 
i=0

def nothing():
    pass
 
cv2.namedWindow("controls") #for live controlling thersold calues

cv2.createTrackbar("lower","controls",50,200,nothing)
cv2.createTrackbar("higher","controls",150,200,nothing)

while True:
    ret,frame=cap.read()
    current_time=time.time()

    taken_time=current_time-previous_time

    fps=1/taken_time

    lower=cv2.getTrackbarPos("lower","controls")
    higher=cv2.getTrackbarPos("higher","controls")
    
    blur_image=cv2.GaussianBlur(frame,(11,11),0)
    gray_image=cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
    canny=cv2.Canny(gray_image,lower,higher)

    if not ret :
        print("webcame loading issue")
        break
    

    key=cv2.waitKey(1) & 0xFF

    if key==ord("e"):
        edge_on= not edge_on
        
        
    elif key==ord("q"):
        print("loop closing sucessfully")
        break
    cv2.putText(blur_image,f"fps:{int(fps)} and time taken in one frame :{taken_time}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,252,0),1)
    if edge_on :
        cv2.imshow("canny image",canny)
    else:
        cv2.imshow("original image",blur_image)

    
    i +=1
    print(f"fps:{int(fps)} and time taken in {i}th frame :{taken_time}")

    # combined = cv2.hconcat([blur_image,canny])
    # cv2.imshow("Output", combined)

    display = canny if edge_on else blur_image
    cv2.imshow("view",display)
    previous_time=current_time


    
cap.release()
cv2.destroyAllWindows()


