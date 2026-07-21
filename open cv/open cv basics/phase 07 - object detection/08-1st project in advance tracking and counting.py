import cv2
from ultralytics import YOLO
import time as time

model=YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)

total_id=set()

pre_time=0

while True :
    ret,frames=cap.read()

    if not ret:
        print("webcame loading issue")
        break

    curr_time=time.time()

    curr_id=set()

    if pre_time==0:
        fps=0
    else :
        fps=1/(curr_time-pre_time)

    results=model.track(frames,persist=True)
    annoted_frame=results[0].plot()



    if  results[0].boxes is not None and results[0].boxes.id is not None:
        for box,track_id in zip(results[0].boxes,results[0].boxes.id):
            cls=int(box.cls[0])

            if cls==0:
                total_id.add(int(track_id))
                curr_id.add(int(track_id))

                x1,y1,x2,y2=map(int,box.xyxy[0])

                cv2.putText(annoted_frame,f"track_id:{track_id} ",(x2,y2-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    cv2.putText(annoted_frame,f"fps:{fps}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,245,0),1)
    cv2.putText(annoted_frame,f"current people:{len(curr_id)}",(15,55),cv2.FONT_HERSHEY_SIMPLEX,1,(0,245,0),2)
    cv2.putText(annoted_frame,f"total people{len(total_id)}",(355,45),cv2.FONT_HERSHEY_SIMPLEX,0.8,(245,0,0,2),2)
    cv2.imshow("person tracking", annoted_frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        print("loop succesfully  closed")
        break

cap.release()
cv2.destroyAllWindows()










