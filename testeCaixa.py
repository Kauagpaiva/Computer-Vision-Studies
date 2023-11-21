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

        # Get the x and y coordinates of the index
        left_index_x, left_index_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].y
        right_index_x, right_index_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y

        # Check if the absolute difference between the x and y coordinates is below the thresholds
        return (abs(left_wrist_x - right_wrist_x) < x_threshold/2 and abs(left_wrist_y - right_wrist_y) < y_threshold/2) or (abs(left_index_x - right_index_x) < x_threshold and abs(left_index_y - right_index_y) < y_threshold)

    return False

def are_hands_inside_box(landmarks, frame, box_center, box_size):
    # Check if both hands are inside the specified box
    draw_square_around_box(frame, box_center, box_size)
    if landmarks is not None and len(landmarks) >= 2:
        # Get the x and y coordinates of the index
        left_wrist_x, left_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX.value].y
        right_wrist_x, right_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y

        # Check if both hands are inside the box
        if (((box_center[0] - box_size/2) < left_wrist_x) and (left_wrist_x < (box_center[0] + box_size/2)) and ((box_center[1] - box_size/2) < left_wrist_y) and (left_wrist_y < (box_center[1] + box_size/2)) and ((box_center[0] - box_size/2) < right_wrist_x) and (right_wrist_x < (box_center[0] + box_size/2)) and ((box_center[1] - box_size/2) < right_wrist_y) and (right_wrist_y < (box_center[1] + box_size/2))):
            print("why it didnt work")
            return True
    print("the code above isnt working")






    
        if (box_center[0] - box_size/2) < left_wrist_x:
            if left_wrist_x < (box_center[0] + box_size/2):
                if (box_center[1] - box_size/2) < left_wrist_y:
                    if left_wrist_y < (box_center[1] + box_size/2):
                        print("it works")
                        return True
                    
    print("the code above isnt working")
    return False

def draw_square_around_box(frame, box_center, box_size):
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2

        # Calculate the half-size of the box
        half_size = box_size // 2

        # Calculate the top-left and bottom-right coordinates of the square
        top_left = (box_center[0] - half_size, box_center[1] - half_size)
        bottom_right = (box_center[0] + half_size, box_center[1] + half_size)

        # Draw the square
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)


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

    # Check if hands are inside the box
    #box_center = (300, 200)
    height, width, _ = frame.shape
    box_center = (height//2, width//2)
    box_size = 350  # Adjust this value based on your preference
    #draw_square_around_box(frame, box_center, box_size)

    if are_hands_inside_box(results.pose_landmarks.landmark,frame, box_center, box_size):
        cv2.putText(frame, "Hands Inside Box", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
