import cv2
import mediapipe as mp
import serial
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()


arduino = serial.Serial('COM5 ', 9600)
time.sleep(2)

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

    if result.multi_hand_landmarks:
        for i, landmarks in enumerate(result.multi_hand_landmarks):
            # Check if it's the left hand using the handedness attribute
            if result.multi_handedness[i].classification[0].label == 'Left':
                # Draw landmarks and connections for the left hand only
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Check finger positions (landmark 4 is the base of the thumb, landmark 8 is the tip of the index finger, etc.)
                fingers_status = []

                # Check for thumb (landmark 4 is thumb base, landmark 3 is thumb tip)
                thumb_open = landmarks.landmark[4].y < landmarks.landmark[3].y
                fingers_status.append(0 if thumb_open else 1)

                # Check for index finger (landmark 6 is index finger base, landmark 8 is index finger tip)
                index_open = landmarks.landmark[6].y < landmarks.landmark[8].y
                fingers_status.append(0 if index_open else 1)

                # Check for middle finger (landmark 10 is middle finger base, landmark 12 is middle finger tip)
                middle_open = landmarks.landmark[10].y < landmarks.landmark[12].y
                fingers_status.append(0 if middle_open else 1)

                # Check for ring finger (landmark 14 is ring finger base, landmark 16 is ring finger tip)
                ring_open = landmarks.landmark[14].y < landmarks.landmark[16].y
                fingers_status.append(0 if ring_open else 1)

                # Check for pinky finger (landmark 18 is pinky base, landmark 20 is pinky tip)
                pinky_open = landmarks.landmark[18].y < landmarks.landmark[20].y
                fingers_status.append(0 if pinky_open else 1)

                # Send the finger status as a string to Arduino
                status_str = " ".join(map(str, fingers_status))
                arduino.write((status_str + "\n").encode())

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
