# Trying to check if both hands are together inside an specific area of the recording, the idea is to use it to verify if someone is or not washing their hands

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

def handsInsideBox(landmarks, center, radius, height, width):
        # Coordinates in percentage (from 0 to 1.00 in x or y)
        center = [center[0]/width, center[1]/height]

        if landmarks is not None and len(landmarks) >= 2:
            # Get the x and y coordinates of the wrists
            left_index_x, left_index_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].y
            right_index_x, right_index_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y

            return ((center[0] - radius) < left_index_x and left_index_x < (center[0] + radius) and 
                    (center[1] - radius) < left_index_y and left_index_y < (center[1] + radius) and 
                    (center[0] - radius) < right_index_x and right_index_x < (center[0] + radius) and 
                    (center[1] - radius) < right_index_y and right_index_y < (center[1] + radius))
        return False

def draw_green_dot_and_rectangle(frame, center, radius, width, height):
    # this coordinates must be in base of the real size of the recording
    # Draw a rectangle around the green dot
    radius_X, radius_Y = width*radius, height*radius
    bottom_left = (int(center[0] - radius_X), int(center[1] - radius_Y))
    top_right = (int(center[0] + radius_X), int(center[1] + radius_Y))

    cv2.rectangle(frame, bottom_left, top_right, (0, 255, 0), 2)  # 2 is the thickness of the rectangle

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

    # Creating a box
    height, width, _ = frame.shape
    center = (width/2, height/2) # X and Y, center of the recording
    radius = 2/10 # 20% of the screensize

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

    # Check if hands are inside the box
    if handsInsideBox(results.pose_landmarks.landmark, center, radius, height, width):
        cv2.putText(frame, "Hands inside the box", (290, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw the box to help us visualize
    draw_green_dot_and_rectangle(frame, center, radius, width, height)

    cv2.imshow('HandWashing V1', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
