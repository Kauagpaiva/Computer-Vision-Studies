import cv2
import mediapipe as mp

# Set global thresholds for the hands to be considered together
x_threshold = 0.1
y_threshold = 0.1

def are_hands_together(landmarks):
    # Check if both wrists are present
    if landmarks is not None and len(landmarks) >= 2:
        # Get the x and y coordinates of the wrists
        left_wrist_x, left_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_x, right_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y

        # Get the x and y coordinates of the wrists
        left_index_x, left_index_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].y
        right_index_x, right_index_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y

        # Check if the absolute difference between the x and y coordinates is below the thresholds
        return (abs(left_wrist_x - right_wrist_x) < x_threshold/2 and abs(left_wrist_y - right_wrist_y) < y_threshold/2) or (abs(left_index_x - right_index_x) < x_threshold and abs(left_index_y - right_index_y) < y_threshold)

    return False

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# OpenCV setup
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw square around each wrist and index
    if results.pose_landmarks is not None:
        #for landmark in [mp.solutions.pose.PoseLandmark.LEFT_WRIST, mp.solutions.pose.PoseLandmark.RIGHT_WRIST]:
            #wrist = results.pose_landmarks.landmark[landmark.value]

            #wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])

            #length = int(x_threshold/2 * frame.shape[1])  # Use x_threshold as a proportion of the frame width
            #cv2.rectangle(frame, (wrist_x - length, wrist_y - length),
                          #(wrist_x + length, wrist_y + length), (0, 255, 0), 2)
            
        for landmark in [mp.solutions.pose.PoseLandmark.LEFT_INDEX, mp.solutions.pose.PoseLandmark.RIGHT_INDEX]:
            index = results.pose_landmarks.landmark[landmark.value]

            index_x, index_y = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])

            length = int(x_threshold * frame.shape[1])  # Use x_threshold as a proportion of the frame width
            cv2.rectangle(frame, (index_x - length, index_y - length),
                          (index_x + length, index_y + length), (0, 255, 0), 2)

    # Check if hands are together
    if are_hands_together(results.pose_landmarks.landmark):
        cv2.putText(frame, "Hands Together", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
