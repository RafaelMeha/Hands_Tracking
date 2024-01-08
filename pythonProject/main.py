import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

circle_radius = 5

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()
while True:
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=circle_radius),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
