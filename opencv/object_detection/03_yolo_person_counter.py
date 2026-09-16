import cv2
from ultralytics import YOLO
import time

model=YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)


pre_time=0
i=0

while True:
    ret,frame=cap.read()

    if not ret:
        break

    cur_time=time.time()

    if pre_time==0:
        fps=0
    else:
        fps=1/(cur_time-pre_time)

    resized_frame=cv2.resize(frame,(640,480))

    result=model(resized_frame)

    annoted_frame=result[0].plot()
    
    person_count=0
    
    for box in result[0].boxes:
        if int(box.cls)==0:
            person_count+=1
    pre_time=cur_time
    i +=1

    print(f"in {i}th loop fps :{fps} and no. of persons are :{person_count}")

    cv2.imshow("view",annoted_frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



