from Hand_gesture_recognition import *
import numpy as np

cap = cv2.VideoCapture(0)
while True:
  _, frame = cap.read()
  frame = cv2.flip(frame, 1)
  frame = hand_gesture_recognition(frame)
  cv2.imshow("Output", frame) 
  if cv2.waitKey(1) == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()
