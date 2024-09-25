# Hisia Tambulizi Flask Backend

This project is a Flask backend that captures real-time video from a webcam, detects faces, and analyzes emotions using the `DeepFace` library. The detected emotions are displayed in real-time, annotated over the video stream.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Routes](#routes)
- [Database Migrations](#database-migrations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Face Detection**: Detects human faces in real-time video streams using OpenCV's Haar Cascade Classifier.
- **Emotion Recognition**: Analyzes faces for emotions such as happy, sad, angry, etc., using `DeepFace`.
- **Web-based Video Feed**: Streams the webcam feed over HTTP, with the ability to view it in a browser.

## Technologies Used

- **Python**: Core programming language.
- **Flask**: Web framework to serve video feed and API.
- **OpenCV**: For video capture and face detection.
- **DeepFace**: For emotion recognition using pre-trained models.
- **TensorFlow**: As required by DeepFace for deep learning models.

## Requirements

Ensure you have Python 3.x installed. The project dependencies include:

- Flask
- OpenCV (`opencv-python-headless` and `opencv-python`)
- DeepFace
- TensorFlow (version 2.x)
- `tf-keras` (if required)

You can install all dependencies using `pip` as outlined in the installation steps below.

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/emotion-detection-backend.git
    cd emotion-detection-backend
    ```

2. **Create and Activate a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install Flask opencv-python deepface tf-keras psycopg2-binary sqlalchemy flask_sqlalchemy flask-cors install bcrypt

    ```
    - Remember to also open and read the `requirements.txt` file.

4. **Download Haar Cascade Classifier**:
    Ensure OpenCV's `haarcascade_frontalface_default.xml` is available by default in your environment. If not, download it from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in your project directory.

## Usage

Once the dependencies are installed, follow these steps to run the project:

1. **Run the Flask Application**:
    ```bash
    python app.py
    ```

    The application will start running on `http://127.0.0.1:5000`.

2. **Access the Video Feed**:
    Open a browser and navigate to `http://127.0.0.1:5000/video_feed` to view the live video feed with face detection and emotion recognition.

## Code Overview

### Main Components

- **`app.py`**: The main Flask application file that handles the video capture, face detection, and emotion recognition.
- **OpenCV (`cv2`)**: Captures video from the default camera, detects faces using the Haar Cascade Classifier, and processes frames for emotion analysis.
- **DeepFace**: Analyzes the faces detected in the video stream and returns the dominant emotion.

### Key Functions

- **`generate_frames()`**:
    - Captures frames from the webcam using OpenCV.
    - Detects faces in each frame using the Haar Cascade Classifier.
    - Analyzes the faces for emotions using `DeepFace.analyze()`.
    - Annotates each detected face with a rectangle and the dominant emotion label.
    - Converts the frame into JPEG format and streams it as a video.

- **`video_feed()`**:
    - Exposes the video feed over HTTP, allowing you to view it in a browser.

### Example Code

Here’s the core logic for face detection and emotion analysis in `video_processing.py` and `app.py`:

```python
# video_processing.py
import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the default camera
    while True:
        success, frame = cap.read()  # Read frame-by-frame
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "Unknown"

            # Draw rectangle around the face and add emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
```
```python
# app.py
from flask import Flask, Response, jsonify
from video_processing import generate_frames

app = Flask(__name__)

@app.route('/')
def welcome():
    return "Welcome to the Emotion Backend API Karibu Sana feel at home"

@app.route('/video_feed')
def video_feed():
    # Return the video feed generated by generate_frames()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
```

## Database Migrations
To manage your database schema with Alembic, follow these steps:

1. Install Alembic:
   ```bash
   pip install alembic
   ```
2. Initialize Alembic: Navigate to your project directory and initialize Alembic:

```bash

alembic init migrations

```
3. Configure Alembic: Open the alembic.ini file and set the sqlalchemy.url to your PostgreSQL URI:

```bash

# alembic.ini
[alembic]
# other settings...

sqlalchemy.url = postgresql://your_username:your_password@localhost/emotion_detection

```

Next, open migrations/env.py and make sure it includes your Flask app and SQLAlchemy db object:
```bash
   # migrations/env.py
from __future__ import with_statement

import logging
from logging.config import fileConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alembic import context
from flask import current_app
from flask_sqlalchemy import SQLAlchemy

# Import your models here
from app import db

# Set up logging
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

target_metadata = db.metadata

def run_migrations_online():
    connectable = create_engine(context.config.get_main_option("sqlalchemy.url"))

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()
```

4. Create a Migration Script:

```bash

alembic revision --autogenerate -m "Initial migration"
```
5. Apply the Migration:

```bash

alembic upgrade head
```

6. Handling Future Migrations: For subsequent schema changes (optional phase you can skip):

 - Make changes to your models.
 - Generate a new migration script:

```bash

alembic revision --autogenerate -m "Describe your changes"

```
7. Apply the new migration (also optional as it communicates about step 6):
```bash

alembic upgrade head
```
## Routes
1.   / :  Takes you to the default welcome page. This route returns a simple text message "Welcome to the Emotion Backend API" to confirm that the server is running and provides an introduction to the application.

2.   /video_feed: This route provides a live video stream captured from your webcam. The video will have rectangles drawn around detected faces and emotion labels displayed on top of each face.


## Troubleshooting
1. DeepFace Import Error: If you encounter ModuleNotFoundError: No module named 'deepface', ensure you have installed DeepFace correctly using pip install deepface.

2. TensorFlow Errors: If you face TensorFlow-related issues, ensure you are using a compatible version. Consider installing tf-keras or downgrading TensorFlow if needed.

3. Video Feed Not Displaying: Ensure your webcam is functioning and accessible by OpenCV.
Check that the URL http://127.0.0.1:5000/video_feed is correct.

## Contributing
Contributions are welcome! If you would like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (git checkout -b feature/new-feature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/new-feature).
5. Open a pull request.

Feel free to report any issues or suggest improvements!

## License
This project is licensed under the MIT License. See the LICENSE file for more information.