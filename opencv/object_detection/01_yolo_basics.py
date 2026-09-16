
from ultralytics import YOLO

model=YOLO("yolov8n.pt")

image=r"C:\Users\dipesh's-device\Pictures\WhatsApp .jpg"

detect=model(image)

detect[0].show()
