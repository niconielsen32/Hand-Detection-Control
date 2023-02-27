import cv2
import mediapipe as mp
import time

from phue import Bridge

b = Bridge('192.168.0.230')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

index_finger_tip = 0
thumb_tip = 0

rescaled_index_finger_tip = (0,0)
rescaled_thumb_tip = (0,0)

distance = 0


cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        success, image = cap.read()

        start = time.time()
        
        
        img_h, img_w, img_c = image.shape
   
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and find hands
        results = hands.process(image)

        image.flags.writeable = True

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            rescaled_index_finger_tip = (int(index_finger_tip.x * img_w), int(index_finger_tip.y * img_h))
            rescaled_thumb_tip = (int(thumb_tip.x * img_w), int(thumb_tip.y * img_h))
            
            distance = ((rescaled_index_finger_tip[0] - rescaled_thumb_tip[0])**2 + (rescaled_index_finger_tip[1] - rescaled_thumb_tip[1])**2)**0.5
            
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
        b.set_light('Kontor', 'bri', int(distance))


        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        
        cv2.line(image, rescaled_index_finger_tip, rescaled_thumb_tip, (0,255,0), 2)

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
          break

cap.release()