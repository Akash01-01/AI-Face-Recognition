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
label_to_emp = {}

# Initialize face cascade with production error handling
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("ERROR: Face cascade failed to load!")
        face_cascade = None
    else:
        print("✓ Face cascade loaded successfully")
except Exception as e:
    print(f"ERROR loading face cascade: {e}")
    face_cascade = None



# New models for employee and attendance

class EmpMaster(db.Model):
    __tablename__ = 'emp_master'
    id = db.Column(db.Integer, primary_key=True)
    emp_id = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(255), default=None)
    dob = db.Column(db.Date, default=None)
    gender = db.Column(db.String(10), default=None)
    email = db.Column(db.String(255), default=None)
    contact = db.Column(db.String(20), default=None)
    present_addr = db.Column(db.Text, default=None)
    perm_addr = db.Column(db.Text, default=None)
    join_date = db.Column(db.Date, default=None)
    end_date = db.Column(db.Date, default=None)
    emp_type = db.Column(db.String(50), default=None)
    check_in = db.Column(db.Time, default=None)
    check_out = db.Column(db.Time, default=None)
    longitude = db.Column(db.Numeric(10,8), default=None)
    latitude = db.Column(db.Numeric(11,8), default=None)
    dept = db.Column(db.String(255), default=None)
    desig = db.Column(db.String(255), default=None)
    salary_type = db.Column(db.String(50), default=None)
    salary_amt = db.Column(db.String(255), nullable=False)
    full_abs_fine = db.Column(db.Numeric(10,2), default=None)
    half_abd_fine = db.Column(db.Numeric(10,2), default=None)
    yearly_leaves = db.Column(db.Integer, default=None)
    bank = db.Column(db.String(255), default=None)
    bank_name = db.Column(db.String(255), default=None)
    branch_name = db.Column(db.String(255), default=None)
    account_name = db.Column(db.String(255), nullable=False)
    account_no = db.Column(db.String(50), default=None)
    ifsc_code = db.Column(db.String(20), default=None)
    entried_by = db.Column(db.String(255), default=None)
    created_date = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    total_yearly_leaves = db.Column(db.String(250), nullable=False)

class EmployeeImage(db.Model):
    __tablename__ = 'employee_images'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    emp_id = db.Column(db.String(255), nullable=False)
    image_data = db.Column(db.LargeBinary(length=16777216), nullable=False)
    filename = db.Column(db.String(255), nullable=True)
    uploaded_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())


class AttendanceMaster(db.Model):
    __tablename__ = 'attendance_master'
    id = db.Column(db.Integer, primary_key=True)
    emp_id = db.Column(db.Integer, nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    check_in = db.Column(db.Time, nullable=False)
    check_out = db.Column(db.Time, default=None)
    worked_hours = db.Column(db.Numeric(5,2), default=None)
    worked_day = db.Column(db.String(20), default=None)
    att_date = db.Column(db.Date, nullable=False)
    longitude = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.String(255), nullable=False)
    is_paid = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    attendance_status = db.Column(db.String(250), nullable=False)

# Face images are now stored directly in the database via the Add Employee form.


# Train recognizer using new employee tables
def train_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_to_emp = {}
    employees = EmpMaster.query.all()
    if not employees:
        return recognizer, label_to_emp
    label = 0
    for emp in employees:
        emp_faces_count = 0
        imgs = EmployeeImage.query.filter_by(emp_id=emp.emp_id).limit(2).all()  # Limit to 2 images per employee
        if not imgs:
            continue
        for img_row in imgs:
            try:
                npbuf = np.frombuffer(img_row.image_data, dtype=np.uint8)
                img = cv2.imdecode(npbuf, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Skip if cascade not available
                if face_cascade is None:
                    print("Skipping face detection in training - cascade not available")
                    continue
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, minSize=(30, 30))
                if len(detected_faces) > 0:
                    # Use only the largest face
                    x, y, w, h = max(detected_faces, key=lambda f: f[2] * f[3])
                    face_roi = img[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (80, 80))  # Even smaller for speed
                    faces.append(face_roi)
                    labels.append(label)
                    emp_faces_count += 1
            except Exception:
                continue
        if emp_faces_count > 0:
            label_to_emp[label] = emp.emp_id  # Store string emp_id
            label += 1
    if faces and labels:
        recognizer.train(faces, np.array(labels))
    return recognizer, label_to_emp


# Helper to get employee info
def get_employee_info(emp_id):
    emp = EmpMaster.query.filter_by(emp_id=emp_id).first()
    if emp:
        # Handle date properly - check if it's a date object and not empty string
        dob_str = 'No DOB'
        if emp.dob and hasattr(emp.dob, 'strftime'):
            dob_str = emp.dob.strftime('%Y-%m-%d')
        elif emp.dob and isinstance(emp.dob, str) and emp.dob.strip():
            dob_str = emp.dob
            
        return {
            'emp_id': emp.emp_id,
            'id': emp.id,  # Integer ID for attendance table
            'name': emp.full_name or emp.emp_id,
            'email': emp.email or 'No email',
            'phone': emp.contact or 'No phone',
            'dob': dob_str
        }
    return {'emp_id': emp_id, 'id': None, 'name': 'Unknown', 'email': 'No email', 'phone': 'No phone', 'dob': 'No DOB'}


def reload_faces():
    global face_recognizer, label_to_emp
    face_recognizer, label_to_emp = train_face_recognizer()


# Fast validation - just check confidence threshold
def validate_confidence(confidence):
    return confidence < 35  # Very strict confidence check only

# Initialize application
with app.app_context():
    db.create_all()
    try:
        # Check OpenCV availability first
        print("Checking OpenCV face recognition...")
        test_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✓ OpenCV face recognition available")
        
        emp_count = EmpMaster.query.count()
        img_count = EmployeeImage.query.count()
        print(f"Found {emp_count} employees and {img_count} images in database")
        
        if emp_count > 0 and face_cascade is not None:
            print("Training face recognizer...")
            face_recognizer, label_to_emp = train_face_recognizer()
            print(f"✓ Trained with {len(label_to_emp)} employees")
        else:
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            label_to_emp = {}
            if face_cascade is None:
                print("⚠ Face cascade not available - face detection disabled")
    except Exception as e:
        print(f"Initialization error: {e}")
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        label_to_emp = {}


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug_info():
    """Debug endpoint to check production status"""
    try:
        emp_count = EmpMaster.query.count()
        img_count = EmployeeImage.query.count()
        return jsonify({
            'employees_in_db': emp_count,
            'images_in_db': img_count,
            'face_recognizer_loaded': face_recognizer is not None,
            'trained_labels': len(label_to_emp),
            'label_mapping': label_to_emp,
            'face_cascade_loaded': not face_cascade.empty() if face_cascade else False
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# Add new employee and images
@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    if request.method == 'POST':
        # Collect all fields from the form
        emp_data = {
            'emp_id': request.form.get('emp_id'),
            'full_name': request.form.get('full_name'),
            'dob': request.form.get('dob'),
            'gender': request.form.get('gender'),
            'email': request.form.get('email'),
            'contact': request.form.get('contact'),
            'present_addr': request.form.get('present_addr'),
            'perm_addr': request.form.get('perm_addr'),
            'join_date': request.form.get('join_date'),
            'end_date': request.form.get('end_date'),
            'emp_type': request.form.get('emp_type'),
            'check_in': request.form.get('check_in'),
            'check_out': request.form.get('check_out'),
            'longitude': request.form.get('longitude'),
            'latitude': request.form.get('latitude'),
            'dept': request.form.get('dept'),
            'desig': request.form.get('desig'),
            'salary_type': request.form.get('salary_type'),
            'salary_amt': request.form.get('salary_amt'),
            'full_abs_fine': request.form.get('full_abs_fine'),
            'half_abd_fine': request.form.get('half_abd_fine'),
            'yearly_leaves': request.form.get('yearly_leaves'),
            'bank': request.form.get('bank'),
            'bank_name': request.form.get('bank_name'),
            'branch_name': request.form.get('branch_name'),
            'account_name': request.form.get('account_name'),
            'account_no': request.form.get('account_no'),
            'ifsc_code': request.form.get('ifsc_code'),
            'entried_by': request.form.get('entried_by'),
            'total_yearly_leaves': request.form.get('total_yearly_leaves'),
        }
        # Convert date/time/number fields
        for date_field in ['dob', 'join_date', 'end_date']:
            if emp_data[date_field] and emp_data[date_field].strip():
                try:
                    emp_data[date_field] = datetime.strptime(emp_data[date_field], '%Y-%m-%d').date()
                except Exception:
                    emp_data[date_field] = None
            else:
                emp_data[date_field] = None
        for time_field in ['check_in', 'check_out']:
            if emp_data[time_field] and emp_data[time_field].strip():
                try:
                    emp_data[time_field] = datetime.strptime(emp_data[time_field], '%H:%M').time()
                except Exception:
                    emp_data[time_field] = None
            else:
                emp_data[time_field] = None
        for float_field in ['longitude', 'latitude', 'full_abs_fine', 'half_abd_fine']:
            if emp_data[float_field] and emp_data[float_field].strip():
                try:
                    emp_data[float_field] = float(emp_data[float_field])
                except Exception:
                    emp_data[float_field] = None
            else:
                emp_data[float_field] = None
        for int_field in ['yearly_leaves']:
            if emp_data[int_field] and emp_data[int_field].strip():
                try:
                    emp_data[int_field] = int(emp_data[int_field])
                except Exception:
                    emp_data[int_field] = None
            else:
                emp_data[int_field] = None
        # Check if employee exists (upsert logic)
        existing_emp = EmpMaster.query.filter_by(emp_id=emp_data['emp_id']).first()
        if existing_emp:
            # Update existing employee
            for key, value in emp_data.items():
                if value:  # Only update non-empty values
                    setattr(existing_emp, key, value)
            db.session.commit()
            emp = existing_emp
        else:
            # Insert new employee
            emp = EmpMaster(**emp_data)
            db.session.add(emp)
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
                    db.session.add(EmployeeImage(emp_id=emp.emp_id, image_data=data, filename=file.filename))
                    added += 1
                except Exception as e:
                    print(f'Error reading uploaded image: {e}')
        if added:
            db.session.commit()
        reload_faces()
        flash('Employee added successfully!')
        return redirect(url_for('add_person'))

@app.route('/retrain')
def force_retrain():
    """Force retrain face recognizer for production"""
    global face_recognizer, label_to_emp
    try:
        old_count = len(label_to_emp)
        face_recognizer, label_to_emp = train_face_recognizer()
        return jsonify({
            'success': True,
            'message': f'Retrained: {old_count} -> {len(label_to_emp)} employees',
            'trained_employees': len(label_to_emp),
            'label_mapping': label_to_emp
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
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

        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if height > 400:
            scale = 400.0 / height
            new_width = int(width * scale)
            frame = cv2.resize(frame, (new_width, 400))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if face cascade is available
        if face_cascade is None:
            print("ERROR: Face cascade not available in production")
            return jsonify({
                'recognized': False,
                'message': 'Face detection service unavailable',
                'debug': 'OpenCV cascade not loaded'
            })
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(50, 50))
        if len(faces) == 0:
            return jsonify({'message': 'No face detected.', 'recognized': False})

        if not label_to_emp:
            return jsonify({
                'recognized': False,
                'message': 'No employees registered yet.',
                'debug': f'label_to_emp empty, face_recognizer: {face_recognizer is not None}'
            })

        # Use only the largest face for maximum speed
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (80, 80))  # Very small for speed
        
        # Single prediction only - no validation needed
        label, confidence = face_recognizer.predict(face_img)
        
        # Debug info for production
        print(f'DEBUG: Prediction - Label: {label}, Confidence: {confidence}')
        print(f'DEBUG: Available employees in label_to_emp: {list(label_to_emp.keys())}')
        
        best_face_crop = None
        if confidence < 50:  # More lenient threshold for production
            # Quick face crop
            try:
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                best_face_crop = frame[y1:y2, x1:x2]
            except Exception:
                pass

        if confidence < 50:  # More lenient for production
            emp_id = label_to_emp.get(label, None)
            print(f'DEBUG: Found emp_id: {emp_id} for label: {label}')
            if emp_id:
                emp_info = get_employee_info(emp_id)
                
                # Quick attendance marking
                today = datetime.utcnow().date()
                current_time = datetime.utcnow().time()
                
                if emp_info['id']:
                    # Ultra-fast attendance check with exists()
                    existing = db.session.query(AttendanceMaster.id).filter_by(emp_id=emp_info['id'], att_date=today).first()
                    if not existing:
                        attendance = AttendanceMaster(
                            emp_id=emp_info['id'],
                            full_name=emp_info['name'],
                            check_in=current_time,
                            att_date=today,
                            longitude='0.0',
                            latitude='0.0',
                            attendance_status='Present'
                        )
                        db.session.add(attendance)
                        db.session.commit()
                
                face_b64 = None
                if best_face_crop is not None:
                    try:
                        _, buf = cv2.imencode('.jpg', best_face_crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        face_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                    except Exception:
                        pass
                
                return jsonify({
                    'recognized': True,
                    'emp_id': emp_info.get('emp_id'),
                    'name': emp_info.get('name'),
                    'email': emp_info.get('email', 'No email'),
                    'phone': emp_info.get('phone', 'No phone'),
                    'dob': emp_info.get('dob', 'No DOB'),
                    'confidence': float(confidence),
                    'face_image': face_b64,
                    'message': f"Welcome {emp_info['name']}! Attendance marked."
                })
        # No valid match found
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


# Production configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
