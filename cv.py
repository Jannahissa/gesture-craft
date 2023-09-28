import cv2
import mediapipe as mp
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Labeling the actions
actions = ['walk', 'break/attack', 'place/use', 'jump', 'look']
seq_length = 20
secs_for_action = 5

# Function to load landmark data for each action
def load_landmark_data(action):
    action_dir = os.path.join(data_path, action.replace('/', '_'))
    landmarks_files = [f for f in os.listdir(action_dir) if f.startswith('landmarks_')]
    action_data = []
    for landmarks_file in landmarks_files:
        landmarks_data = np.load(os.path.join(action_dir, landmarks_file))
        action_data.extend(landmarks_data)
    return np.array(action_data)

cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands.Hands()

created_time = int(time.time())
desktop_path = '/Users/jannah/Desktop/actions' 
os.makedirs(desktop_path, exist_ok=True)

# Load the recorded landmark data for each action
data_path = desktop_path
landmarks_break_attack = load_landmark_data('break/attack')
landmarks_place_use = load_landmark_data('place/use')
landmarks_walk = load_landmark_data('walk')
landmarks_jump = load_landmark_data('jump')
landmarks_look = load_landmark_data('look')

# Combine the loaded landmark data into X and y arrays
X = np.concatenate([landmarks_walk, landmarks_break_attack, landmarks_place_use, landmarks_jump, landmarks_look])
y = np.concatenate([
    np.zeros(landmarks_walk.shape[0]),
    np.ones(landmarks_break_attack.shape[0]),
    np.ones(landmarks_place_use.shape[0]) * 2,
    np.ones(landmarks_jump.shape[0]) * 3,
    np.ones(landmarks_look.shape[0]) * 4
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Train MLP 
mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train, y_train)

# Train KNN 
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Recognition using the classifiers
while True:
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

            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            input_data = np.array([d])

            # Predict the action using SVM 
            svm_prediction = int(svm_classifier.predict(input_data)[0])
            action_svm = actions[svm_prediction]

            # Predict the action using MLP 
            mlp_prediction = int(mlp_classifier.predict(input_data)[0])
            action_mlp = actions[mlp_prediction]

            # Predict the action using KNN 
            knn_prediction = int(knn_classifier.predict(input_data)[0])
            action_knn = actions[knn_prediction]

            cv2.putText(frame, f'SVM: {action_svm}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'MLP: {action_mlp}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'KNN: {action_knn}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
