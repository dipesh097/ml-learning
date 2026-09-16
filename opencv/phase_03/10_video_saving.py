import cv2

cap=cv2.VideoCapture(0)

image_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec=cv2.VideoWriter_fourcc(*"XVID")        
recorder=cv2.VideoWriter("output.avi",codec,20,(image_width,image_height))

i=0

while True:
    ret , frame=cap.read()
    i +=1

    if ret is None:
        print("loading issue")
        break
    print(f"loop no. {i+1}")
    recorder.write(frame)
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF== ord("Q"):
        print("loop closing succesfully")
        break

cap.release()
recorder.release()
cv2.destroyAllWindows()

    
