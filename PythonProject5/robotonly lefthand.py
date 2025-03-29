import cv2
import mediapipe as mp
import serial
import time

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()


# Initialize serial communication with Arduino (adjust port as needed)
arduino = serial.Serial('COM5', 9600)  # Replace 'COM3' with your Arduino port
time.sleep(2)  # Wait for the serial connection to initialize

# Set up the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    result = hands.process(rgb_frame)

    # Variable to track if a left hand is found
    left_hand_found = False

    if result.multi_hand_landmarks and result.multi_handedness:
        for i, landmarks in enumerate(result.multi_hand_landmarks):
            # Check if it's the left hand using the handedness attribute
            if result.multi_handedness[i].classification[0].label == 'Left':
                if not left_hand_found:  # Only process the first detected left hand
                    left_hand_found = True  # Mark the left hand as found

                    # Draw landmarks and connections for the left hand only
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    # Check finger positions
                    fingers_status = []

                    # Thumb (landmark 4 is thumb base, landmark 3 is thumb tip)
                    thumb_open = landmarks.landmark[4].y < landmarks.landmark[3].y
                    fingers_status.append(0 if thumb_open else 1)

                    # Index finger (landmark 6 is index finger base, landmark 8 is index finger tip)
                    index_open = landmarks.landmark[6].y < landmarks.landmark[8].y
                    fingers_status.append(0 if index_open else 1)

                    # Middle finger (landmark 10 is middle finger base, landmark 12 is middle finger tip)
                    middle_open = landmarks.landmark[10].y < landmarks.landmark[12].y
                    fingers_status.append(0 if middle_open else 1)

                    # Ring finger (landmark 14 is ring finger base, landmark 16 is ring finger tip)
                    ring_open = landmarks.landmark[14].y < landmarks.landmark[16].y
                    fingers_status.append(0 if ring_open else 1)

                    # Pinky finger (landmark 18 is pinky base, landmark 20 is pinky tip)
                    pinky_open = landmarks.landmark[18].y < landmarks.landmark[20].y
                    fingers_status.append(0 if pinky_open else 1)

                    # Send the finger status as a string to Arduino
                    status_str = " ".join(map(str, fingers_status))
                    arduino.write((status_str + "\n").encode())
                break  # Exit the loop after processing the first left hand

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
