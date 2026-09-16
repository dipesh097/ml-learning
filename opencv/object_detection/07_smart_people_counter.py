import cv2
from ultralytics import YOLO
import time

model=YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)

pre_time=0

while True:
    ret,frame=cap.read()

    if not ret:
        break

    cur_time=time.time()

    fps=1/(cur_time-pre_time)

    result=model(frame)

    annotated_frame=result[0].plot()
    
    person_count=0

    for box in result[0].boxes:
        if box.cls==0:
            person_count +=1
    
    if person_count<=3:
        cv2.putText(annotated_frame,f"fps:{fps}, total person :{person_count} \n so there is not crowed",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    else:
        cv2.putText(annotated_frame,f"fps:{fps}, total person :{person_count} \n so there is crowed",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,466,655),2)


    

    cv2.imshow("live",annotated_frame)

    pre_time=cur_time

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


    