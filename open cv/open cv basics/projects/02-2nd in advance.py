import cv2
from ultralytics import YOLO
from deepface import DeepFace

model=YOLO("yolov8n.pt")

reference=r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20260409_01_28_04_Pro.jpg"

cap=cv2.VideoCapture(0)


face_detecter=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret,frame=cap.read()
    if not ret:
        print("webcame loading issue")
        break

    results=model.track(frame,persist=True)
    annoted_frame=results[0].plot()



    if results[0].boxes and results[0].boxes.id is not None:
        for box,track_id in zip(results[0].boxes , results[0].boxes.id):
            cls=int(box[0].cls)

            if cls==0:

                    x1,y1,x2,y2=map(int,box.xyxy[0])
                    new_frame=frame[y1:y2,x1:x2]import cv2





                    face = face_detecter.detectMultiScale(new_frame, 1.3, 5)

                    for x,y,w,h in face :
                        cv2.rectangle(new_frame,(x,y),((x+w),(y+h)),(255,5,234),1)

                        detected_face=new_frame[y:y+h,x:x+w]


                        try:
                            result=DeepFace.verify(
                                 img1=detected_face,
                                 img2=reference,
                                 enforce_detection=False
                                     )

                            if result["verified"]:
                                 text="face are matched"
                            else:
                                 text="face are not matched"

                            cv2.putText(frame, text, (50, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 266, 0), 1)
                        except:
                                  print("error occurs")
                                  break


            cv2.imshow("detection",frame)



        if cv2.waitKey(1) & 0xFF==ord("q"):
            print("loop successfully closed")
            break

cap.release()
cv2.destroyAllWindows()



