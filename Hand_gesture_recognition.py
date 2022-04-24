import cv2
import mediapipe as mp
from tensorflow.python.keras.models import load_model
import numpy as np

mpHands = mp.solutions.hands 
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7) 
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer pretrained model
model = load_model(r"../project/mp_hand_gesture")
f = open(r"../project/gesture.names", 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# The model can recognize 10 different gestures.
def hand_gesture_recognition(frame):
  x , y, c = frame.shape
  framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
  result = hands.process(framergb)

  # post process the result
  if result.multi_hand_landmarks:
    landmarks = []
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            lmx = int(lm.x * x)
            lmy = int(lm.y * y)
            landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    if landmarks!=[]:
      prediction = model.predict([landmarks])
      classID = np.argmax(prediction)
      className = classNames[classID]
      # show the prediction on the frame
      cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
      
  return frame

