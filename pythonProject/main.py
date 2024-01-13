import cv2
import mediapipe as mp

gesture_emoji_map = {
    'index_up': '☝️',
    'full_open_hand': '🖐️',
    'call_me': '🤙',
    'middle_finger': '🖕',
    'thumbs_up': '👍',
    'thumbs_down': '👎',
}

def is_finger_up(finger_tip, finger_dip, hand_landmarks):
    return hand_landmarks.landmark[finger_tip].y < \
           hand_landmarks.landmark[finger_dip].y

def is_full_open_hand(hand_landmarks):
    return all(is_finger_up(mp_hands.HandLandmark[tip], mp_hands.HandLandmark[dip], hand_landmarks)
               for tip, dip in [('THUMB_TIP', 'THUMB_IP'), ('INDEX_FINGER_TIP', 'INDEX_FINGER_DIP'),
                               ('MIDDLE_FINGER_TIP', 'MIDDLE_FINGER_DIP'), ('RING_FINGER_TIP', 'RING_FINGER_DIP'),
                               ('PINKY_TIP', 'PINKY_DIP')])
def is_index_finger_up(hand_landmarks):
    index_up = is_finger_up(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, hand_landmarks)
    other_fingers_down = all(not is_finger_up(mp_hands.HandLandmark[tip], mp_hands.HandLandmark[dip], hand_landmarks)
                             for tip, dip in [('MIDDLE_FINGER_TIP', 'MIDDLE_FINGER_DIP'),
                                             ('RING_FINGER_TIP', 'RING_FINGER_DIP'),
                                             ('PINKY_TIP', 'PINKY_DIP')])
    return index_up and other_fingers_down

def is_call_me(hand_landmarks):
    thumb_up = is_finger_up(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP, hand_landmarks)
    pinky_up = is_finger_up(mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP, hand_landmarks)
    other_fingers_down = not any(is_finger_up(mp_hands.HandLandmark[tip], mp_hands.HandLandmark[dip], hand_landmarks)
                                 for tip, dip in [('INDEX_FINGER_TIP', 'INDEX_FINGER_DIP'),
                                                 ('MIDDLE_FINGER_TIP', 'MIDDLE_FINGER_DIP'),
                                                 ('RING_FINGER_TIP', 'RING_FINGER_DIP')])
    return thumb_up and pinky_up and other_fingers_down
def is_middle_finger(hand_landmarks):
    middle_up = is_finger_up(mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, hand_landmarks)
    other_fingers_down = all(not is_finger_up(mp_hands.HandLandmark[tip], mp_hands.HandLandmark[dip], hand_landmarks)
                             for tip, dip in [('THUMB_TIP', 'THUMB_IP'),
                                             ('INDEX_FINGER_TIP', 'INDEX_FINGER_DIP'),
                                             ('RING_FINGER_TIP', 'RING_FINGER_DIP'),
                                             ('PINKY_TIP', 'PINKY_DIP')])
    return middle_up and other_fingers_down

def is_thumbs_up(hand_landmarks):
    thumb_up = is_finger_up(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP, hand_landmarks)
    other_fingers_down = all(not is_finger_up(mp_hands.HandLandmark[tip], mp_hands.HandLandmark[dip], hand_landmarks)
                             for tip, dip in [('INDEX_FINGER_TIP', 'INDEX_FINGER_DIP'),
                                             ('MIDDLE_FINGER_TIP', 'MIDDLE_FINGER_DIP'),
                                             ('RING_FINGER_TIP', 'RING_FINGER_DIP'),
                                             ('PINKY_TIP', 'PINKY_DIP')])
    return thumb_up and other_fingers_down

def is_thumbs_down(hand_landmarks):
    thumb_down = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > \
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    other_fingers_down = all(not is_finger_up(mp_hands.HandLandmark[tip], mp_hands.HandLandmark[dip], hand_landmarks)
                             for tip, dip in [('INDEX_FINGER_TIP', 'INDEX_FINGER_DIP'),
                                             ('MIDDLE_FINGER_TIP', 'MIDDLE_FINGER_DIP'),
                                             ('RING_FINGER_TIP', 'RING_FINGER_DIP'),
                                             ('PINKY_TIP', 'PINKY_DIP')])
    return thumb_down and other_fingers_down

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=5)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    results_hands = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec, connection_drawing_spec)

            if is_full_open_hand(hand_landmarks):
                print(gesture_emoji_map['full_open_hand'])
            elif is_call_me(hand_landmarks):
                print(gesture_emoji_map['call_me'])
            elif is_middle_finger(hand_landmarks):
                print(gesture_emoji_map['middle_finger'])
            elif is_index_finger_up(hand_landmarks):
                print(gesture_emoji_map['index_up'])
            elif is_thumbs_up(hand_landmarks):
                print(gesture_emoji_map['thumbs_up'])
            elif is_thumbs_down(hand_landmarks):
                print(gesture_emoji_map['thumbs_down'])

    cv2.imshow('Hands Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()