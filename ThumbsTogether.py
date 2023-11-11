import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def detect_thumbs_together(frame):
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
        # If at least two hands are detected, check if both hands are close together
        landmarks_hand1 = results.multi_hand_landmarks[0].landmark
        landmarks_hand2 = results.multi_hand_landmarks[1].landmark

        # Calculate the distance between the landmarks of both hands (e.g., the middle fingertips)
        thumb_tip_x1 = landmarks_hand1[4].x
        thumb_tip_y1 = landmarks_hand1[4].y
        thumb_tip_x2 = landmarks_hand2[4].x
        thumb_tip_y2 = landmarks_hand2[4].y

        distance = ((thumb_tip_x1 - thumb_tip_x2)**2 + (thumb_tip_y1 - thumb_tip_y2)**2)**0.5

        if distance < 0.1:  # You can adjust the distance threshold
            return True  # Hands are close together
        else:
            return False  # Hands are not close together
    else:
        return False  # Not enough hands detected

def main():
    cap = cv2.VideoCapture(0)  # Open the webcam

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect hands and check if they are close together
        are_hands_together = detect_thumbs_together(frame)

        if are_hands_together:
            cv2.putText(frame, "Hands Together", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw hand landmarks if available
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the result
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
