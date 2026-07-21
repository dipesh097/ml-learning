import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Load model
model = YOLO("yolov8n.pt")

# Reference image
reference = r"C:\Users\dipesh's-device\Pictures\Camera Roll\WIN_20260409_01_28_04_Pro.jpg"

# Webcam
cap = cv2.VideoCapture(0)

# Haar cascade
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Track identity memory
identity_map = {}   # track_id → "Known"/"Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam issue")
        break

    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:

        for box, track_id in zip(results[0].boxes, results[0].boxes.id):
            cls = int(box.cls)

            if cls == 0:  # person class
                track_id = int(track_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Ensure valid crop
                if x2 <= x1 or y2 <= y1:
                    continue

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # Convert to grayscale for Haar
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                # Draw person box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # If already recognized, skip heavy computation
                if track_id in identity_map:
                    label = identity_map[track_id]
                else:
                    label = "Detecting..."

                    if len(faces) > 0:
                        # Take first face
                        x, y, w, h = faces[0]

                        face_crop = person_crop[y:y+h, x:x+w]

                        if face_crop.size == 0:
                            continue

                        try:
                            result_verify = DeepFace.verify(
                                img1=face_crop,
                                img2=reference,
                                enforce_detection=False
                            )

                            if result_verify["verified"]:
                                label = "AUTHORIZED"
                            else:
                                label = "UNKNOWN"

                            # Save identity for this track_id
                            identity_map[track_id] = label

                        except:
                            label = "ERROR"

                # Draw label
                cv2.putText(frame, f"ID {track_id}: {label}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

    cv2.imshow("AI Face System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()