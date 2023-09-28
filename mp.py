import cv2
import mediapipe as mp
import numpy as np
import time
import os

# labeling the actions
actions = ['walk', 'break/attack', 'place/use', 'jump', 'look']
seq_length = 20
secs_for_action = 5

cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands.Hands()

created_time = int(time.time())
desktop_path = '/Users/jannah/Desktop/actions'  
os.makedirs(desktop_path, exist_ok=True)

for idx, action in enumerate(actions):
    data = []
    start_time = time.time()

    # Collecting action data
    while (time.time() - start_time) < 3:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Collecting {action.upper()} action', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('frame', frame)
        key = cv2.waitKeyEx(1)
        if key != -1:
            break

    start_time = time.time()

    while (time.time() - start_time) < secs_for_action:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        key = cv2.waitKeyEx(1)
        if key == ord('q'):
            break

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))
                    cv2.putText(frame, str(len(landmarks) - 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Compute angles between joints
                joint = np.zeros((21, 4))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]
                angle = np.degrees(angle)  # Convert radian to degree

                # Check for jump action
                THRESHOLD_ANGLE = 45  # Adjust this value as needed
                if any(angle > THRESHOLD_ANGLE for angle in angle[3:6]):
                    print('Jump action detected!')

                # Check for look action
                if angle[8] > THRESHOLD_ANGLE:
                    print('Look action detected!')

                d = np.concatenate([joint.flatten(), angle])
                data.append(d)

                drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        time.sleep(0.1)

    data = np.array(data)
    print(action, data.shape)

    action_dir = os.path.join(desktop_path, action.replace('/', '_'))
    os.makedirs(action_dir, exist_ok=True)

    # Save the landmarks data for the action
    np.save(os.path.join(action_dir, f'landmarks_{action.replace("/", "_")}_{created_time}.npy'), data)

    # Create data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)

    # Save the sequence data for the action
    np.save(os.path.join(action_dir, f'seq_{action.replace("/", "_")}_{created_time}.npy'), full_seq_data)

cap.release()
cv2.destroyAllWindows()
