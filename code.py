!pip install pandas opencv-python numpy scikit-learn joblib
import pandas as pd

# Assuming 'data.csv' is in the current working directory
data = pd.read_csv('data.csv')
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

data = pd.read_csv('data.csv')

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to load image at {image_path}")
        return None
    image_resized = cv2.resize(image, (64, 64))
    return image_resized.flatten()

image_paths = data['location'].values
X_images = []
valid_ages = []
valid_genders = []

for img_path, age, gender in zip(image_paths, data['age'].values, data['gender'].values):
    img = preprocess_image(img_path)
    if img is not None:
        X_images.append(img)
        valid_ages.append(age)
        valid_genders.append(1 if gender == 'Male' else 0)

X_images = np.array(X_images)
y = np.array(valid_ages)
gender = np.array(valid_genders)
X = np.hstack((X_images, gender.reshape(-1, 1)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

joblib.dump(model, 'age_detection_model.pkl')
import numpy as np
import cv2
import joblib

try:
    model = joblib.load('age_detection_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    if image.size == 0:
        print("Warning: Empty image received.")
        return None
    image_resized = cv2.resize(image, (64, 64))
    return image_resized.flatten()

default_gender = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_preprocessed = preprocess_image(face_roi)

        if face_preprocessed is not None:
            X_features = np.hstack((face_preprocessed, default_gender)).reshape(1, -1)
            predicted_age = model.predict(X_features)[0]
            predicted_age = max(predicted_age, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {int(predicted_age)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Age Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
import joblib

try:
    model = joblib.load('age_detection_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    if image.size == 0:
        print("Warning: Empty image received.")
        return None
    image_resized = cv2.resize(image, (64, 64))
    return image_resized.flatten()

default_gender = 1
final_predicted_age = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_preprocessed = preprocess_image(face_roi)

        if face_preprocessed is not None:
            X_features = np.hstack((face_preprocessed, default_gender)).reshape(1, -1)
            predicted_age = model.predict(X_features)[0]
            predicted_age = max(predicted_age, 0)
            final_predicted_age = predicted_age  # Store the last predicted age
            
            # Draw rectangle and put predicted age on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Age: {int(predicted_age)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Age Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

# Print the final predicted age after the video capture ends
if final_predicted_age is not None:
    print(f'Final Predicted Age: {int(final_predicted_age)}')
else:
    print("No face detected, final predicted age not available.")
