import cv2

# Initialize the HOG descriptor and set the SVM detector to the default people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_full_body(video_source=0):
    # Open the video source (default is the webcam)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        # Read a frame from the video source
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        # Resize the frame to improve detection performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Detect people in the frame
        # Returns the bounding boxes and weights of detected people
        boxes, weights = hog.detectMultiScale(frame_resized,
                                              winStride=(8, 8),
                                              padding=(8, 8),
                                              scale=1.05)

        # Draw bounding boxes around detected people
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Full Body Detection', frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Pass 0 to use the webcam or provide a video file path
    detect_full_body()
