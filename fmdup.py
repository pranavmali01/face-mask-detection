import cv2

# Load the pre-trained face and mask classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mask_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mask.xml')

# Function to detect face masks
def detect_mask(frame, gray_frame):
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Region of Interest (ROI) for face detection
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Detect masks in the face ROI
        masks = mask_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        
        # Draw rectangles around faces and masks
        for (mx, my, mw, mh) in masks:
            cv2.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 255, 0), 3)
            cv2.putText(frame, 'Mask', (x+mx, y+my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, 'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face masks
    frame = detect_mask(frame, gray_frame)
    
    # Display the output
    cv2.imshow('Face Mask Detection', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
