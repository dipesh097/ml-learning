from locale import currency
import cv2

from ultralytics import YOLO
import time as time


cap=cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")

i=0 # for printing no. of loops

pre_time=0

while True:

    ret,frame=cap.read()
    curr_time=time.time()
    i +=1
    
    if pre_time==0:
        fps=0
    else:
        fps=1/(curr_time-pre_time)
    time_taken=(curr_time-pre_time)

    if not ret:
        print("webcame loading issue")
        break

    result=model(frame)

    annoted_frame=result[0].plot()

    cv2.imshow("detected objects",annoted_frame)

    print(f"fps:{fps} in {i}th loop and time taken is {time_taken} ")

    pre_time=curr_time

    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()