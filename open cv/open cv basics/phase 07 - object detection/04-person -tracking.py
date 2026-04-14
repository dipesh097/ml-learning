import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")

tracking=False
tracker=None

frame_count=0

cap=cv2.VideoCapture(0)

while True:
    ret , frame=cap.read()

    if not ret:
        break

    frame_count +=1

    if not tracking or frame_count % 40 ==0:
        results=model(frame)

        for box in results[0].boxes:
            if int(box.cls) ==0:
                x1,y1,x2,y2=map(int,box.xyxy[0])

                bbox=(x1,y1,(x2-x1),(y2-y1))

                tracker=cv2.TrackerCSRT_create()
                tracker.init(frame,bbox)
                tracking=True
                break
    
    if tracking and tracker is not None:
        success,bbox=tracker.update(frame)
        
        x,y,w,h=map(int,bbox)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(256,256,256),1)
        cv2.putText(frame,"tracking",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    else :
        tracking =False
    

    cv2.imshow("tracking live",frame)

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

cap.release()
cv2.destroyAllWindows()        

