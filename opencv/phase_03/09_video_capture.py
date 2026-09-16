import  cv2

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if ret is None:
        print("issue in loading frame")
        break
    print("frame loading successfully")
    cv2.rectangle(frame,(10,50),(500,550),(0,256,256),10)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) &  0xFF==ord("p"):
        print("successfully close loop")
        break
cap.release()
cv2.destroyAllWindows()