import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)

in_counter=set()
out_counter=set()
track_history={}


while True:
    ret,frame=cap.read()

    if not ret:
        print("webcame loading issue")

    results=model.track(frame,persist=True)
    annoted_frame=results[0].plot()

    cv2.line(annoted_frame,(0,220),(1000,220),(255,0,0),2)
    line_y=220

    if results[0].boxes is not None and  results[0].boxes.id is not None:
        for box,track_id in zip(results[0].boxes,results[0].boxes.id):
            cls=int(box.cls[0])

            if cls==0:
                track_id=int(track_id)

                x1,y1,x2,y2=map(int,box.xyxy[0])
                curr_y=(y1+y2)/2

                if track_id in track_history:
                    pre_y=track_history[track_id]

                    if pre_y>line_y and curr_y<line_y :
                        out_counter.add(track_id)
                    elif pre_y<line_y and curr_y>line_y:
                        in_counter.add(track_id)

                track_history[track_id]=curr_y


    cv2.putText(annoted_frame,f"total entries:{len(in_counter)}",(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),3)
    cv2.putText(annoted_frame, f"total exits:{len(out_counter)}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 255), 3)
    cv2.imshow("tracking",annoted_frame)



    if cv2.waitKey(1) & 0xFF==ord("q"):
        print("loop successfully closed")
        break

cap.release()
cv2.destroyAllWindows()