from flask import Flask, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from models import db, User
from video_processing import generate_frames
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app)
db.init_app(app)
bcrypt = Bcrypt(app)

@app.route('/')
def welcome():
    return "Welcome to the Emotion Backend API. Karibu Sana, feel at home."


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User already exists"}), 400

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "User created successfully"}), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "An error occurred. Please try again."}), 500

@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Sign in successful"}), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401
    
@app.route('/video_feed')
def video_feed():
    # Your video processing function
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
