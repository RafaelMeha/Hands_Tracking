import cv2
import mediapipe as mp

gesture_emoji_map = {
    'index_up': '☝️',
}
def is_index_finger_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

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
                mp_hands.HAND_CONNECTIONS)

            if is_index_finger_up(hand_landmarks):
                print(gesture_emoji_map['index_up'])

    cv2.imshow('Hands Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
