import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def detect_hands_together(frame):
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # If multiple hands are detected, check if they are close together
        landmarks = results.multi_hand_landmarks[0].landmark

        # Calculate the Euclidean distance between the middle fingertips of both hands
        thumb_tip_x = landmarks[4].x
        thumb_tip_y = landmarks[4].y
        index_tip_x = landmarks[8].x
        index_tip_y = landmarks[8].y

        distance = ((thumb_tip_x - index_tip_x)**2 + (thumb_tip_y - index_tip_y)**2)**0.5

        if distance < 0.1:  # You can adjust the distance threshold
            return True  # Hands are close together
        else:
            return False  # Hands are not close together
    else:
        return False  # No hands detected

def main():
    cap = cv2.VideoCapture(0)  # Open the webcam

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect hands and check if they are close together
        are_hands_together = detect_hands_together(frame)

        if are_hands_together:
            cv2.putText(frame, "Hand closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
