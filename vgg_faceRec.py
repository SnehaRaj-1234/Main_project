import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('familiar_face_recognition_model.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess the captured frame (resize, normalize, etc.)
    face_resized = cv2.resize(frame, (224, 224))  # Resize the frame to match the model input size
    face_normalized = face_resized / 255.0  # Normalize pixel values to [0, 1]
    face_input = np.expand_dims(face_normalized, axis=0)  # Add batch dimension for prediction

    # Get prediction (familiar or unfamiliar)
    prediction = model.predict(face_input)
    label = 'Familiar' if prediction >= 0.5 else 'Unfamiliar'

    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)  # Show the frame with the label

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
