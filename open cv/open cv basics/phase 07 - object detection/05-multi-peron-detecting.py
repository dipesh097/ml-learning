import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")

trackers=[]
trackers_id=[]
next_id=0

frame_count=0

cap=cv2.VideoCapture(0)

while True :
    ret,frame=cap.read()

    frame_count +=1

    if not ret:
        break

    if frame_count%30==0:
        results=model(frame)

        tracker=[]
        trackers_id=[]
    
        for box in results[0].boxes:
            if int(box.cls)==0:
                x1,y1,x2,y2=map(int,box.xyxy[0])

                bbox=(x1,y1,(x2-x1),(y2-y1))

                tracker=cv2.TrackerCSRT_create()
                tracker.init(frame,bbox)

                trackers.append(tracker)
                trackers_id.append(next_id)
                next_id +=1
    
    for i , tracker in enumerate(trackers):
        success,bbox=tracker.update(frame)

        x,y,w,h=map(int,bbox)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow("live detection",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


