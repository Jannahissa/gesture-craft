import cv2
import mediapipe as mp
import numpy as np
import time
import os
import mcpi.minecraft as minecraft 
import pyautogui 
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from mcpi.minecraft import Minecraft
from mcpi import block


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



# Train KNN 
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)


# Initialize Minecraft connection
try:
    mc = minecraft.Minecraft.create(address="localhost", port=4711)  
except Exception as e:
    print("Failed to connect to the Minecraft server:", e)
    exit()

player_pos = mc.player.getTilePos()



def move_forward():
    direction_pos = mc.player.getDirection()
    yaw_rad = math.atan2(direction_pos.x, direction_pos.z)
    pitch_rad = math.asin(-direction_pos.y)
    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    step_size = 1

    movement_x = step_size * math.sin(math.radians(yaw_deg))
    # movement_y = -step_size * math.sin(math.radians(pitch_deg)) 
    movement_y = step_size * math.sin(math.radians(pitch_deg)) 
    movement_z = step_size * math.cos(math.radians(yaw_deg))
    new_player_x = mc.player.getPos().x + movement_x
    new_player_y = mc.player.getPos().y + movement_y
    new_player_z = mc.player.getPos().z + movement_z
    mc.player.setPos(new_player_x, new_player_y, new_player_z)

def break_block():
    direction_pos = mc.player.getDirection()
    yaw_rad = math.atan2(direction_pos.x, direction_pos.z)
    pitch_rad = math.asin(-direction_pos.y) 
    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    step_size = 1.0

    block_pos_x = int(round(mc.player.getPos().x + step_size * math.sin(math.radians(yaw_deg))))
    block_pos_y = int(round(mc.player.getPos().y + step_size * math.sin(math.radians(pitch_deg))))
    block_pos_z = int(round(mc.player.getPos().z + step_size * math.cos(math.radians(yaw_deg))))
    mc.setBlock(block_pos_x, block_pos_y, block_pos_z, block.AIR)

def place_block():
    direction_pos = mc.player.getDirection()
    yaw_rad = math.atan2(direction_pos.x, direction_pos.z)
    pitch_rad = math.asin(-direction_pos.y)
    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    step_size = 1.0

    block_pos_x = int(round(mc.player.getPos().x + step_size * math.sin(math.radians(yaw_deg))))
    block_pos_y = int(round(mc.player.getPos().y + step_size * math.sin(math.radians(pitch_deg))))
    block_pos_z = int(round(mc.player.getPos().z + step_size * math.cos(math.radians(yaw_deg))))
    mc.setBlock(block_pos_x, block_pos_y, block_pos_z, block.STONE)



def look_right():
    player_pos = mc.player.getPos()
    current_yaw = mc.player.getRotation()
    new_yaw = current_yaw + 45
    mc.player.setRotation(new_yaw)
    
try:
    player_pos = mc.player.getPos()
    
except mcpi.connection.RequestError as e:
    print(f"Error while getting player position: {e}")



def look_left():
    player_pos = mc.player.getPos()
    current_yaw = mc.player.getRotation()
    new_yaw = current_yaw - 45
    mc.player.setRotation(new_yaw)
    
def jump():
    step_size = 1
    player_pos = mc.player.getPos()
    new_player_y = player_pos.y + step_size
    mc.player.setPos(player_pos.x, new_player_y, player_pos.z)


    player_pos = mc.player.getTilePos()
    player_pos.y += 1
    mc.player.setTilePos(player_pos.x, player_pos.y, player_pos.z)
    
    
previous_landmarks = None
    

while True:
    current_action = ''
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    key = cv2.waitKeyEx(1)
    if key == ord('q'):
        break
#    landmark_diffs = [(previous_landmarks[i].x - results.multi_hand_landmarks[i].x, previous_landmarks[i].y - results.multi_hand_landmarks[i].y) for i in range(len(results.multi_hand_landmarks))  ]
    
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

            # Predict the action using KNN
            knn_prediction = int(knn_classifier.predict(input_data)[0])
            action_knn = actions[knn_prediction]

            # Perform the action only if it is different from the current action
            if action_knn != current_action:
                if action_knn == 'walk':
                    move_forward()
                elif action_knn == 'break/attack':
                    break_block()
                elif action_knn == 'place/use':
                    place_block()
                elif action_knn == 'look':
                    if previous_landmarks is not None:
                        for prev_landmarks in previous_landmarks:
                                landmarks_diff = []
                                c_index = 0
                                for prev_l in prev_landmarks.landmark:
                                    diff_x = landmarks[c_index][0] - prev_l.x
                                    diff_y = landmarks[c_index][1] - prev_l.y
                                    landmarks_diff.append((diff_x, diff_y))
                                    c_index += 1
                                print(landmarks_diff)
                                
                                x_coordinate = prev_landmarks.landmark[0].x
                                if x_coordinate > 700:
                                    look_right()
                                    time.sleep(.5)
                                else:
                                    look_left()
                                    time.sleep(.5)
                                #Compare the differences in x-axis (diff_x) to decide whether to look left or right
                                total_diff_x = sum(diff_x for diff_x, _ in landmarks_diff)
                                x_coordinate = prev_landmarks.landmark[0].x
                                if total_diff_x > 0:
                                   look_right()
                                else:
                                   look_left()
                elif action_knn == 'jump':
                    jump()
                    time.sleep(0.5)

                # Update the current action
                current_action = action_knn

                # Debugging: Print the actions and predictions
                print("Action KNN:", action_knn)
                print("Current Action:", current_action)

            cv2.putText(frame, f'KNN: {action_knn}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            

            
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    previous_landmarks = results.multi_hand_landmarks

cap.release()
cv2.destroyAllWindows()
