import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        break

    result=model.track(frame,persist=True,tracker="bytetrack.yaml",classes=[39])

    anotated_frame=result[0].plot()

    cv2.imshow("live tracking with byte track",anotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


