import os
from datetime import datetime
from flask_cors import CORS
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

# Production configuration
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_super_secret_key_here_change_in_production')
# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://u673831287_qa_attdb:p*NsybHiq0V@92.113.22.3/u673831287_qa_attdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable to save memory
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,
    'pool_timeout': 20,
    'max_overflow': 0
}
CORS(app)
db = SQLAlchemy(app)

# Cache for face recognizer
face_recognizer = None
label_to_name = {}


# Person model
class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    dob = db.Column(db.Date, nullable=True)
    folder = db.Column(db.String(120), nullable=False)
    # Images are stored in the database; folder kept for backward-compatibility
    images = db.relationship('PersonImage', backref='person', lazy=True, cascade='all, delete-orphan')


class PersonImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=True)
    image_data = db.Column(db.LargeBinary(length=16777216), nullable=False)  # up to 16MB
    created_at = db.Column(db.DateTime, nullable=False, server_default=db.func.now())

# Note: We no longer persist captured images. Images for training
# must be added manually under 'known_person/<PersonName>/' and
# metadata in the database via the Add Person form.

# Optimized face recognizer training
def train_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_to_name = {}
    people = Person.query.all()
    
    if not people:
        return recognizer, label_to_name
        
    label = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for person in people:
        person_faces_count = 0
        imgs = PersonImage.query.filter_by(person_id=person.id).all()
        if not imgs:
            continue
        for img_row in imgs:
            try:
                npbuf = np.frombuffer(img_row.image_data, dtype=np.uint8)
                img = cv2.imdecode(npbuf, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                if len(detected_faces) == 0:
                    # Fallback detection
                    detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20))
                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    face_roi = cv2.equalizeHist(face_roi)
                    face_roi = cv2.resize(face_roi, (200, 200))
                    face_roi = cv2.normalize(face_roi, None, 0, 255, cv2.NORM_MINMAX)
                    sharpness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                    if sharpness > 10:
                        faces.append(face_roi)
                        labels.append(label)
                        person_faces_count += 1
            except Exception:
                continue
        if person_faces_count > 0:
            label_to_name[label] = person.name
            label += 1
    
    if faces and labels:
        recognizer.train(faces, np.array(labels))
    return recognizer, label_to_name

# Optimized helper functions
def get_person_info(name):
    person = Person.query.filter_by(name=name).first()
    if person:
        return {
            'name': person.name, 
            'email': person.email or 'No email',
            'phone': getattr(person, 'phone', None) or 'No phone',
            'dob': person.dob.strftime('%Y-%m-%d') if hasattr(person, 'dob') and person.dob else 'No DOB'
        }
    return {'name': name, 'email': 'No email', 'phone': 'No phone', 'dob': 'No DOB'}

def reload_faces():
    global face_recognizer, label_to_name
    face_recognizer, label_to_name = train_face_recognizer()

def validate_against_stored_images(captured_face, person_name, confidence):
    try:
        person = Person.query.filter_by(name=person_name).first()
        if not person:
            return False
        imgs = PersonImage.query.filter_by(person_id=person.id).all()
        if not imgs:
            return False
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        similar_matches = 0
        total_comparisons = 0
        
        for img_row in imgs:
            npbuf = np.frombuffer(img_row.image_data, dtype=np.uint8)
            stored_img = cv2.imdecode(npbuf, cv2.IMREAD_GRAYSCALE)
            if stored_img is None:
                continue
            stored_faces = face_cascade.detectMultiScale(stored_img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            
            for (x, y, w, h) in stored_faces:
                stored_face = stored_img[y:y+h, x:x+w]
                stored_face = cv2.equalizeHist(stored_face)
                stored_face = cv2.resize(stored_face, (200, 200))
                stored_face = cv2.normalize(stored_face, None, 0, 255, cv2.NORM_MINMAX)
                
                # Multiple similarity calculations
                similarities = []
                result1 = cv2.matchTemplate(captured_face, stored_face, cv2.TM_CCOEFF_NORMED)
                _, max_val1, _, _ = cv2.minMaxLoc(result1)
                similarities.append(max_val1)
                
                result2 = cv2.matchTemplate(captured_face, stored_face, cv2.TM_CCORR_NORMED)
                _, max_val2, _, _ = cv2.minMaxLoc(result2)
                similarities.append(max_val2)
                
                diff = cv2.absdiff(captured_face, stored_face)
                structural_sim = 1.0 - (np.mean(diff) / 255.0)
                similarities.append(structural_sim)
                
                avg_similarity = np.mean(similarities)
                total_comparisons += 1
                
                if avg_similarity > 0.55:
                    similar_matches += 1
        
        if total_comparisons > 0:
            similarity_ratio = similar_matches / total_comparisons
            return similarity_ratio >= 0.3 and confidence < 95
        return False
        
    except Exception:
        return False

# Initialize application
with app.app_context():
    db.create_all()
    try:
        people_count = Person.query.count()
        if people_count > 0:
            face_recognizer, label_to_name = train_face_recognizer()
        else:
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            label_to_name = {}
    except Exception as e:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        label_to_name = {}


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        dob = request.form['dob']
        folder = name  # keep for compatibility

        # Upsert person record
        person = Person.query.filter_by(name=name).first()
        if not person:
            dob_date = None
            if dob:
                try:
                    dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                except ValueError:
                    flash(f'Invalid date format. Please use YYYY-MM-DD format.')
                    return redirect(url_for('add_person'))
            person = Person(name=name, email=email, phone=phone, dob=dob_date, folder=folder)
            db.session.add(person)
            db.session.commit()
        else:
            # Update optional fields
            person.email = email
            person.phone = phone
            try:
                person.dob = datetime.strptime(dob, '%Y-%m-%d').date() if dob else person.dob
            except ValueError:
                flash(f'Invalid date format. Please use YYYY-MM-DD format.')
                return redirect(url_for('add_person'))
            db.session.commit()

        # Save uploaded images into DB
        files = request.files.getlist('images')
        added = 0
        for idx, file in enumerate(files):
            if file and file.filename:
                try:
                    data = file.read()
                    if not data:
                        continue
                    db.session.add(PersonImage(person_id=person.id, filename=file.filename, image_data=data))
                    added += 1
                except Exception as e:
                    print(f'Error reading uploaded image: {e}')
        if added:
            db.session.commit()

        reload_faces()
        flash('Person added successfully!')
        return redirect(url_for('add_person'))
    return render_template('add_person.html')

@app.route('/identify', methods=['POST'])
def identify():
    try:
        file = request.files.get('image')
        if file is None:
            return jsonify({'error': 'No image provided', 'recognized': False}), 400

        img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image', 'recognized': False}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(50, 50))
        if len(faces) == 0:
            return jsonify({'message': 'No face detected.', 'recognized': False})

        if not label_to_name:
            face_b64 = None
            try:
                x, y, w, h = faces[0]
                frame_h, frame_w = frame.shape[:2]
                expand_factor = 0.5
                padding_x = int(w * expand_factor)
                padding_y = int(h * expand_factor)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(frame_w, x + w + padding_x)
                y2 = min(frame_h, y + h + int(padding_y * 1.5))
                color_crop = frame[y1:y2, x1:x2]
                _, buf = cv2.imencode('.jpg', color_crop)
                face_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            except Exception as e:
                print(f"Error preparing face image: {e}")
            return jsonify({
                'recognized': False,
                'message': 'No people trained yet. Add people first.',
                'face_image': face_b64
            })

        STRICT_CONFIDENCE_THRESHOLD = 80
        MODERATE_CONFIDENCE_THRESHOLD = 100  # not used; kept for backward compatibility
        recognizers = [face_recognizer]
        best_match = None
        best_confidence = float('inf')
        best_face_crop = None
        all_predictions = []

        # Optimized face processing
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            try:
                # Streamlined preprocessing
                face_img = cv2.equalizeHist(face_img)
                face_img = cv2.resize(face_img, (200, 200))
                face_img_normalized = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)
                
                # Multiple predictions for accuracy
                predictions = []
                label1, conf1 = face_recognizer.predict(face_img_normalized)
                predictions.append((label1, conf1))
                
                face_img_contrast = cv2.convertScaleAbs(face_img_normalized, alpha=1.2, beta=10)
                label2, conf2 = face_recognizer.predict(face_img_contrast)
                predictions.append((label2, conf2))
                
                # Find best prediction
                best_prediction = min(predictions, key=lambda x: x[1])
                label, confidence = best_prediction
                
                if confidence < best_confidence:
                    best_confidence = confidence
                    if confidence < STRICT_CONFIDENCE_THRESHOLD:
                        best_match = label
                        # Save face crop
                        try:
                            frame_h, frame_w = frame.shape[:2]
                            expand_factor = 0.5
                            padding_x = int(w * expand_factor)
                            padding_y = int(h * expand_factor)
                            x1 = max(0, x - padding_x)
                            y1 = max(0, y - padding_y)
                            x2 = min(frame_w, x + w + padding_x)
                            y2 = min(frame_h, y + h + int(padding_y * 1.5))
                            best_face_crop = frame[y1:y2, x1:x2]
                        except Exception:
                            best_face_crop = None
            except Exception:
                continue

        if best_match is not None and best_confidence < STRICT_CONFIDENCE_THRESHOLD:
            name = label_to_name.get(best_match, 'Unknown')
            validation_passed = validate_against_stored_images(face_img, name, best_confidence)
            if validation_passed:
                person_info = get_person_info(name)
                face_b64 = None
                if best_face_crop is not None:
                    try:
                        _, buf = cv2.imencode('.jpg', best_face_crop)
                        face_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                    except Exception as e:
                        print(f"Error encoding face crop: {e}")
                return jsonify({
                    'recognized': True,
                    'name': person_info.get('name'),
                    'email': person_info.get('email', 'No email'),
                    'phone': person_info.get('phone', 'No phone'),
                    'dob': person_info.get('dob', 'No DOB'),
                    'confidence': float(best_confidence),
                    'face_image': face_b64,
                    'message': f"Identified: {person_info['name']} [Confidence: {best_confidence:.2f}]"
                })
        elif best_match is not None and best_confidence < MODERATE_CONFIDENCE_THRESHOLD:
            name = label_to_name.get(best_match, 'Unknown')
            person_info = get_person_info(name)
            face_b64 = None
            if best_face_crop is not None:
                try:
                    _, buf = cv2.imencode('.jpg', best_face_crop)
                    face_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                except Exception as e:
                    print(f"Error encoding face crop: {e}")
            return jsonify({
                'recognized': True,
                'name': person_info.get('name'),
                'email': person_info.get('email', 'No email'),
                'phone': person_info.get('phone', 'No phone'),
                'dob': person_info.get('dob', 'No DOB'),
                'confidence': float(best_confidence),
                'face_image': face_b64,
                'message': f"Identified: {person_info['name']} [Confidence: {best_confidence:.2f}] (Moderate match)"
            })
        return jsonify({
            'recognized': False,
            'message': 'Detection unsuccessful.'
        })
    except Exception as e:
        import traceback
        print('Error in /identify:', e)
        traceback.print_exc()
        return jsonify({
            'recognized': False,
            'message': 'Internal server error. Please try again.',
            'error': str(e),
            'face_image': None
        }), 500

@app.route('/register_unknown', methods=['POST'])
def register_unknown():
    return jsonify({'message': 'Unknown person registration is disabled.'}), 403

    

# Production configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
