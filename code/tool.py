import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
LEFT_HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
RIGHT_HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
SCALE_FACTOR = 1.0
SHOULDER_INDICES = [11, 12]

def compute_distances(landmarks, connections):
    distances = []
    for connection in connections:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        # Compute Euclidean distance
        distance = np.linalg.norm(np.array([start_point[0], start_point[1], start_point[2]]) -
                                  np.array([end_point[0], end_point[1], end_point[2]]))
        distances.append(distance)
    return distances

# def extract_keypoints(results):
#     # Existing feature extraction
#     if results.pose_landmarks:
#         pose_landmarks = results.pose_landmarks.landmark
#         print(pose_landmarks)
#         pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_landmarks]).flatten()
#         pose_distances = compute_distances(pose_landmarks, POSE_CONNECTIONS)
#     else:
#         pose = np.zeros(33 * 4)
#         pose_distances = np.zeros(len(POSE_CONNECTIONS))
#
#     if results.left_hand_landmarks:
#         left_hand_landmarks = results.left_hand_landmarks.landmark
#         lh = np.array([[res.x, res.y, res.z] for res in left_hand_landmarks]).flatten()
#         lh_distances = compute_distances(left_hand_landmarks, LEFT_HAND_CONNECTIONS)
#     else:
#         lh = np.zeros(21 * 3)
#         lh_distances = np.zeros(len(LEFT_HAND_CONNECTIONS))
#
#     if results.right_hand_landmarks:
#         right_hand_landmarks = results.right_hand_landmarks.landmark
#         rh = np.array([[res.x, res.y, res.z] for res in right_hand_landmarks]).flatten()
#         rh_distances = compute_distances(right_hand_landmarks, RIGHT_HAND_CONNECTIONS)
#     else:
#         rh = np.zeros(21 * 3)
#         rh_distances = np.zeros(len(RIGHT_HAND_CONNECTIONS))
#
#         # Proximity Features
#     proximity_features = []
#
#     if results.pose_landmarks and results.left_hand_landmarks:
#         # Example: Distance between left wrist and nose
#         left_wrist = results.left_hand_landmarks.landmark[0]  # Assuming index 0 is wrist
#         nose = results.pose_landmarks.landmark[0]  # Assuming index 0 is nose
#         distance = np.linalg.norm(
#             np.array([left_wrist.x, left_wrist.y, left_wrist.z]) -
#             np.array([nose.x, nose.y, nose.z])
#         )
#         proximity_features.append(distance)
#     else:
#         proximity_features.append(0.0)
#
#     if results.pose_landmarks and results.right_hand_landmarks:
#         # Example: Distance between right wrist and nose
#         right_wrist = results.right_hand_landmarks.landmark[0]  # Assuming index 0 is wrist
#         nose = results.pose_landmarks.landmark[0]  # Assuming index 0 is nose
#         distance = np.linalg.norm(
#             np.array([right_wrist.x, right_wrist.y, right_wrist.z]) -
#             np.array([nose.x, nose.y, nose.z])
#         )
#         proximity_features.append(distance)
#     else:
#         proximity_features.append(0.0)
#
#
#     # Concatenate all features
#     keypoints = np.concatenate([pose, lh, rh])
#     connection_features = np.concatenate([pose_distances, lh_distances, rh_distances])
#     proximity_features = np.array(proximity_features)
#
#     # Final Feature Vector
#     return np.concatenate([keypoints, connection_features, proximity_features])

def extract_keypoints2(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh,rh])
def extract_keypoints(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """
    Extract keypoints from normalized landmarks.

    Args:
        pose_landmarks: Normalized pose landmarks as a numpy array of shape (33, 4).
        left_hand_landmarks: Normalized left hand landmarks as a numpy array of shape (21, 3).
        right_hand_landmarks: Normalized right hand landmarks as a numpy array of shape (21, 3).

    Returns:
        np.ndarray: Concatenated feature vector.
    """
    # Extract pose keypoints
    if pose_landmarks is not None:
        pose = pose_landmarks.flatten()  # Flatten to 1D array
        pose_distances = compute_distances(pose_landmarks, POSE_CONNECTIONS)
    else:
        pose = np.zeros(33 * 4)
        pose_distances = np.zeros(len(POSE_CONNECTIONS))

    # Extract left hand keypoints
    if left_hand_landmarks is not None:
        lh = left_hand_landmarks.flatten()  # Flatten to 1D array
        lh_distances = compute_distances(left_hand_landmarks, LEFT_HAND_CONNECTIONS)
    else:
        lh = np.zeros(21 * 3)
        lh_distances = np.zeros(len(LEFT_HAND_CONNECTIONS))

    # Extract right hand keypoints
    if right_hand_landmarks is not None:
        rh = right_hand_landmarks.flatten()  # Flatten to 1D array
        rh_distances = compute_distances(right_hand_landmarks, RIGHT_HAND_CONNECTIONS)
    else:
        rh = np.zeros(21 * 3)
        rh_distances = np.zeros(len(RIGHT_HAND_CONNECTIONS))

    # Proximity Features
    proximity_features = []

    if pose_landmarks is not None and left_hand_landmarks is not None:
        left_wrist = left_hand_landmarks[0]  # index 0 is wrist
        nose = pose_landmarks[0]  #  index 0 is nose
        distance = np.linalg.norm(left_wrist[:3] - nose[:3])  # Use only x, y, z
        proximity_features.append(distance)
    else:
        proximity_features.append(0.0)

    if pose_landmarks is not None and right_hand_landmarks is not None:
        right_wrist = right_hand_landmarks[0] # index 0 is nose
        nose = pose_landmarks[0]  # index 0 is nose
        distance = np.linalg.norm(right_wrist[:3] - nose[:3])  # Use only x, y, z
        proximity_features.append(distance)
    else:
        proximity_features.append(0.0)

    # Concatenate all features
    keypoints = np.concatenate([pose, lh, rh])
    connection_features = np.concatenate([pose_distances, lh_distances, rh_distances])
    proximity_features = np.array(proximity_features)

    # Final Feature Vector
    return np.concatenate([keypoints, connection_features, proximity_features])
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
class PoseNormalizer:
    @staticmethod
    def normalize_holistic(
            pose_landmarks: np.ndarray,
            left_hand_landmarks: np.ndarray,
            right_hand_landmarks: np.ndarray,
            left_shoulder_idx: int = SHOULDER_INDICES[0],
            right_shoulder_idx: int = SHOULDER_INDICES[1],
            scale_factor: float = SCALE_FACTOR
    ) -> tuple:
        """
        Normalizes holistic landmarks (pose, left hand, right hand) based on shoulder distance and centers the midpoint.

        Args:
            pose_landmarks: Input array of shape (33, 4) containing (x, y, z, visibility)
            left_hand_landmarks: Input array of shape (21, 3) containing (x, y, z)
            right_hand_landmarks: Input array of shape (21, 3) containing (x, y, z)
            left_shoulder_idx: Index of left shoulder landmark
            right_shoulder_idx: Index of right shoulder landmark
            scale_factor: Desired distance between shoulders after normalization

        Returns:
            Normalized pose, left hand, and right hand landmarks
        """
        # Extract shoulder landmarks
        l_shoulder = pose_landmarks[left_shoulder_idx, :3]
        r_shoulder = pose_landmarks[right_shoulder_idx, :3]

        # Calculate midpoint and shoulder distance
        midpoint = (l_shoulder + r_shoulder) / 2.0
        shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)

        # Normalize pose landmarks
        pose_landmarks[:, :3] -= midpoint
        scale = scale_factor / shoulder_dist if shoulder_dist > 0 else 1.0
        pose_landmarks[:, :3] *= scale

        # Normalize left hand landmarks
        if left_hand_landmarks is not None:
            left_hand_landmarks -= midpoint
            left_hand_landmarks *= scale

        # Normalize right hand landmarks
        if right_hand_landmarks is not None:
            right_hand_landmarks -= midpoint
            right_hand_landmarks *= scale

        return pose_landmarks, left_hand_landmarks, right_hand_landmarks

def calculate_palm_normal(hand_landmarks, handedness_label):
    # Get the three key points for the palm plane
    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    # Calculate vectors
    if handedness_label == 'Right':
        vector1 = index_mcp - wrist
        vector2 = pinky_mcp - wrist
    else:
        vector1 = pinky_mcp - wrist
        vector2 = index_mcp - wrist

    normal_vector = np.cross(vector1, vector2)
    norm = np.linalg.norm(normal_vector)
    if norm > 1e-8:
        normal_vector = normal_vector / norm
    else:
        normal_vector = np.zeros(3)
    return normal_vector

def classify_hand_view(normal_vector, handedness_label):
    nx, ny, nz = normal_vector
    threshold = 0.7
    side_threshold = 0.3

    if nz > threshold:
        return [1,0,0,0,0]
    elif nz < -threshold:
        return [0,1,0,0,0]
    else:
        if handedness_label == 'Right':
            if nx > side_threshold:
                return [0,0,1,0,0]
            elif nx < -side_threshold:
                return [0,0,0,1,0]
        else:
            if nx > side_threshold:
                return [0,0,0,1,0]
            elif nx < -side_threshold:
                return [0,0,1,0,0]
        return [0,0,0,0,1]

def calculate_hand_rotation(hand_landmarks):
    # Middle finger MCP joint (landmark 9)
    mmcp = hand_landmarks[9]
    # Wrist joint (landmark 0)
    wrist = hand_landmarks[0]

    # Calculate the vector from wrist to middle finger MCP
    vector = mmcp - wrist
    delta_x = vector[0]
    delta_y = vector[1]

    # Calculate the angle in degrees
    angle = math.degrees(math.atan2(delta_y, delta_x))
    angle = (angle + 360) % 360

    # Rotation bucket
    bucket = int(((angle + 22.5) % 360) / 45) % 8

    one_hot_bucket = [0] * 8
    one_hot_bucket[bucket] = 1

    return one_hot_bucket

def determine_hand_shape(hand_landmarks):
    """
    Determines the shape of the hand based on finger landmarks.
    """
    # Function to calculate the angle between three points
    def calculate_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    # Function to check if a finger is extended
    def is_finger_extended(mcp, pip, dip, tip):
        angle_pip = calculate_angle(mcp, pip, dip)
        angle_dip = calculate_angle(pip, dip, tip)
        return angle_pip > 135 and angle_dip > 135  # Adjust thresholds as needed

    # Function to check if the thumb is extended
    def is_thumb_extended(thumb_cmc, thumb_mcp, thumb_ip, thumb_tip):
        angle_mcp = calculate_angle(thumb_cmc, thumb_mcp, thumb_ip)
        angle_ip = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        return angle_mcp > 135 and angle_ip > 135  # Adjust thresholds as needed

    # Get landmarks for each finger
    thumb_cmc = hand_landmarks[1]
    thumb_mcp = hand_landmarks[2]
    thumb_ip = hand_landmarks[3]
    thumb_tip = hand_landmarks[4]

    index_mcp = hand_landmarks[5]
    index_pip = hand_landmarks[6]
    index_dip = hand_landmarks[7]
    index_tip = hand_landmarks[8]

    middle_mcp = hand_landmarks[9]
    middle_pip = hand_landmarks[10]
    middle_dip = hand_landmarks[11]
    middle_tip = hand_landmarks[12]

    ring_mcp = hand_landmarks[13]
    ring_pip = hand_landmarks[14]
    ring_dip = hand_landmarks[15]
    ring_tip = hand_landmarks[16]

    pinky_mcp = hand_landmarks[17]
    pinky_pip = hand_landmarks[18]
    pinky_dip = hand_landmarks[19]
    pinky_tip = hand_landmarks[20]

    # Determine if each finger is extended
    thumb_extended = is_thumb_extended(thumb_cmc, thumb_mcp, thumb_ip, thumb_tip)
    index_extended = is_finger_extended(index_mcp, index_pip, index_dip, index_tip)
    middle_extended = is_finger_extended(middle_mcp, middle_pip, middle_dip, middle_tip)
    ring_extended = is_finger_extended(ring_mcp, ring_pip, ring_dip, ring_tip)
    pinky_extended = is_finger_extended(pinky_mcp, pinky_pip, pinky_dip, pinky_tip)

    extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

    # Define hand shapes based on which fingers are extended
    if all(extended_fingers):
        return [1,0,0,0,0,0,0] # Open Hand
    elif not any(extended_fingers):
        return [0,1,0,0,0,0,0] # Fist
    elif index_extended and middle_extended and not any([ring_extended, pinky_extended]):
        return [0,0,1,0,0,0,0] # Victory
    elif index_extended and thumb_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return [0,0,0,1,0,0,0] # Pointing
    elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return [0,0,0,0,1,0,0] # Thumb Up
    elif index_extended and thumb_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return [0,0,0,0,0,1,0] # Rock On
    else:
        return [0,0,0,0,0,0,1] # Other Shape

def calculate_finger_angle(mcp, pip, tip):
    """
    Calculate the angle between three finger landmarks (MCP, PIP, TIP) and return a one-hot encoded bucket.

    Args:
        mcp: Metacarpophalangeal joint landmark as a numpy array of shape (3,).
        pip: Proximal interphalangeal joint landmark as a numpy array of shape (3,).
        tip: Tip of the finger landmark as a numpy array of shape (3,).

    Returns:
        tuple: A tuple containing the angle in degrees and the bucket index.
    """
    # Calculate vectors in 3D (fix: was 2D only, missed depth-oriented fingers)
    vector1 = pip[:3] - mcp[:3]  # [x, y, z]
    vector2 = tip[:3] - pip[:3]  # [x, y, z]

    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    # Calculate the angle in degrees (fix: clamp to prevent acos domain error)
    if magnitude < 1e-8:
        angle = 0  # Handle division by zero
    else:
        cosine = np.clip(dot_product / magnitude, -1.0, 1.0)
        angle = math.degrees(math.acos(cosine))

    # Calculate the bucket (0 to 7)
    bucket = int(((angle + 22.5) % 360) / 45) % 8

    # One-hot encode the bucket
    one_hot_bucket = [0] * 8
    one_hot_bucket[bucket] = 1

    return angle, bucket

def one_hot_finger(hand_landmarks):
    fingers = {
        "Thumb": (hand_landmarks[1], hand_landmarks[2], hand_landmarks[4]),
        "Index": (hand_landmarks[5], hand_landmarks[6], hand_landmarks[8]),
        "Middle": (hand_landmarks[9], hand_landmarks[10], hand_landmarks[12]),
        "Ring": (hand_landmarks[13], hand_landmarks[14], hand_landmarks[16]),
        "Pinky": (hand_landmarks[17], hand_landmarks[18], hand_landmarks[20]),
    }
    all_finger_buckets = []
    for j, (finger, (mcp, pip, tip)) in enumerate(fingers.items()):
        angle, bucket = calculate_finger_angle(mcp,pip,tip)
        one_hot_bucket = [0] * 8
        one_hot_bucket[bucket] = 1
        all_finger_buckets.extend(one_hot_bucket)
    # print("-------------")
    # for i in range(0, len(all_finger_buckets), 8):
    #     print(all_finger_buckets[i:i + 8])
    # print("-------------")
    return all_finger_buckets
def display_hand_details(frame, hand_landmarks, handedness_label, palm_orientation, rotation_angle, rotation_bucket, hand_shape, h, w):
    """ Function to display hand details such as angles, palm orientation, rotation, etc. """

    # Finger Joints
    fingers = {
        "Thumb": (hand_landmarks[1], hand_landmarks[2], hand_landmarks[4]),
        "Index": (hand_landmarks[5], hand_landmarks[6], hand_landmarks[8]),
        "Middle": (hand_landmarks[9], hand_landmarks[10], hand_landmarks[12]),
        "Ring": (hand_landmarks[13], hand_landmarks[14], hand_landmarks[16]),
        "Pinky": (hand_landmarks[17], hand_landmarks[18], hand_landmarks[20]),
    }

    # Calculate Angles and Display Text
    y_offset = 30
    for j, (finger, (mcp, pip, tip)) in enumerate(fingers.items()):
        vector1 = pip - mcp
        vector2 = tip - pip
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        angle = math.degrees(math.acos(dot_product / magnitude)) if magnitude != 0 else 0
        bucket = int(((angle + 22.5) % 360) / 45) % 8
        text_x, text_y = int(pip[0] * w), int(pip[1] * h) - 10 * j
        cv2.putText(frame, f"{finger}: {angle:.1f}° (B{bucket})", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display Handedness, Shape, Palm Orientation, and Rotation
    cv2.putText(frame, f'Hand: {handedness_label}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Shape: {hand_shape}', (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Palm: {palm_orientation}', (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Rotation: {rotation_angle:.1f}° (Bucket {rotation_bucket})', (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw landmarks and connections
    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

