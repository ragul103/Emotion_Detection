import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pygame

# Initialize pygame
pygame.init()

# Function to get music recommendation based on emotion
def get_music_recommendation(emotion):
    emotion_music_mapping = {
        'Angry':[r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"],
        'Disgust': [r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"],
        'Fear': [r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"],
        'Happy': [r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"],
        'Sad': [r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"],
        'Surprise': [r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"],
        'Neutral': [r"C:\Users\rglra\Desktop\Emotion_Detection\badassma.mp3"]
    }
    return emotion_music_mapping.get(emotion, ['No Recommendation'])

# Function for real-time facial emotion recognition from camera
def recognize_emotion_from_camera():
    # Open a connection to the camera (0 indicates the default camera)
    cap = cv2.VideoCapture(0)

    # Load the model and specify the optimizer
    emotion_model = load_model(r"C:\Users\rglra\Desktop\Emotion_Detection\emotion_model.hdf5", compile=False, custom_objects={'Adam': Adam()})

    # Create a Pygame mixer object
    pygame.mixer.init()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Perform face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi / 255.0  # Normalize the image

            # Reshape the image to fit the model input
            face_roi = np.reshape(face_roi, (1, 64, 64, 1))

            # Predict emotion
            emotion_prediction = emotion_model.predict(face_roi)
            emotion_index = np.argmax(emotion_prediction)
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion = emotion_labels[emotion_index]

            # Get music recommendation
            music_recommendation = get_music_recommendation(emotion)

            # Display the emotion and music recommendation on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Music Recommendation: {', '.join(music_recommendation)}", (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Play the recommended song
            for song_path in music_recommendation:
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()

        # Display the resulting frame
        cv2.imshow('Facial Emotion Recognition', frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run real-time facial emotion recognition from the camera
recognize_emotion_from_camera()
