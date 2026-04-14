from turtle import color
import cv2
from deepface import DeepFace

reference=r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20251225_12_27_05_Pro (2).jpg"

face_detecter=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

while True :
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_detecter.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x+y),((x+w),(y+h)),(0,255,0),2)

        face_crop=frame[y:y+h,x:x+h]

        try :
            result=DeepFace.verify(
                img1_path=face_crop,
                img2_path=reference,
                enforce_detection=False
            )

            if result["varified"]:

                text="Face match"
                color=(0,255,0)
            else :
                text="Face not match"
                colour=(0,0,255)
        except:
            text="Error"
            color=(0,255,255)
        cv2.putText(frame,text,(x,y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,
          color,2)

    cv2.imshow("face verifiction",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()









