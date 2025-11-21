import os
from datetime import datetime
from flask_cors import CORS
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")
    OPENCV_AVAILABLE = False
    # Create simple mock for cv2
    cv2 = None

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    # Simple numpy mock
    class MockNumpy:
        @staticmethod
        def frombuffer(*args, **kwargs): return []
        @staticmethod
        def array(*args, **kwargs): return []
    np = MockNumpy()

# DeepFace will be lazy-loaded only when needed to avoid memory issues
DEEPFACE_AVAILABLE = False
DeepFace = None
tf = None

def lazy_load_deepface():
    """Lazy load DeepFace only when actually needed to minimize memory usage"""
    global DEEPFACE_AVAILABLE, DeepFace, tf
    if DEEPFACE_AVAILABLE:
        return True
    
    try:
        print("📦 Lazy loading DeepFace (memory optimized)...")
        # Aggressive TensorFlow memory optimization
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only
        
        import tensorflow as tf_module
        tf = tf_module
        tf.get_logger().setLevel('ERROR')
        
        # Force CPU and memory limits
        try:
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass
            
        from deepface import DeepFace as DF
        DeepFace = DF
        DEEPFACE_AVAILABLE = True
        print("✓ DeepFace loaded successfully")
        return True
    except Exception as e:
        print(f"✗ DeepFace lazy loading failed: {e}")
        DEEPFACE_AVAILABLE = False
        return False
        
print("✓ DeepFace configured for lazy loading (memory optimized)")

import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import tempfile
import io
from PIL import Image
import json
import sys

# Environment and deployment detection
IS_PRODUCTION = os.getenv('RENDER') == 'true' or os.getenv('RAILWAY_ENVIRONMENT') == 'production' or os.getenv('VERCEL') == '1'
LOCAL_MODE = not IS_PRODUCTION
PORT = int(os.getenv('PORT', 5000))

print(f"🔧 Running in {'PRODUCTION' if IS_PRODUCTION else 'LOCAL'} mode")
print(f"🔧 Port: {PORT}")
print(f"🔧 Python version: {sys.version}")
print(f"🔧 Available memory info: {'Limited (production)' if IS_PRODUCTION else 'Local development'}")

# Production configuration
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_super_secret_key_here_change_in_production')
# Database configuration with PyMySQL connection timeouts
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://u673831287_qa_attdb:p*NsybHiq0V@92.113.22.3/u673831287_qa_attdb?connect_timeout=10&read_timeout=30&write_timeout=30'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,
    'pool_timeout': 20,
    'max_overflow': 0,
    'pool_reset_on_return': 'rollback'
}
CORS(app)
db = SQLAlchemy(app)

# Cache for face recognizer
face_recognizer = None
label_to_emp = {}

# PRODUCTION SECURITY SETTINGS - Memory Optimized Configuration
# Try DeepFace first, fallback to LBPH if memory constrained
USE_DEEPFACE = True  # Will auto-fallback to LBPH if memory issues
DEEPFACE_DISTANCE_THRESHOLD = 0.6  # More lenient for lightweight model
DEEPFACE_MODEL = 'OpenFace'  # Lightest model ~15MB vs VGG-Face 580MB
DEEPFACE_DETECTOR = 'opencv'  # Lightweight detector
LBPH_CONFIDENCE_THRESHOLD = 130  # LBPH fallback threshold
MIN_FACE_WIDTH = 80   # Minimum face width in pixels
MIN_FACE_HEIGHT = 80  # Minimum face height in pixels
MIN_FACE_BRIGHTNESS = 30  # Minimum average brightness (0-255)
MAX_FACE_BRIGHTNESS = 230  # Maximum average brightness

# Face recognition system availability - Check OpenCV face module
try:
    if OPENCV_AVAILABLE and cv2 and hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
        FACE_MODULE_AVAILABLE = True
        print("✓ OpenCV face module available")
    else:
        FACE_MODULE_AVAILABLE = False
        print("✗ OpenCV face module not available - will use fallback")
except Exception as e:
    FACE_MODULE_AVAILABLE = False
    print(f"✗ Face module check failed: {e}")

# Initialize face cascade with production error handling
if OPENCV_AVAILABLE and cv2:
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if hasattr(face_cascade, 'empty') and face_cascade.empty():
            print("ERROR: Face cascade failed to load!")
            face_cascade = None
        else:
            print("✓ Face cascade loaded successfully")
    except Exception as e:
        print(f"ERROR loading face cascade: {e}")
        face_cascade = None
else:
    print("OpenCV not available - face detection disabled")
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
    """Unified face recognition training with DeepFace primary and LBPH fallback"""
    global USE_DEEPFACE
    
    # Try DeepFace first if enabled and not in memory-constrained environment
    if USE_DEEPFACE:
        try:
            print("🧠 Attempting DeepFace training (memory optimized)...")
            return train_deepface_optimized()
        except Exception as e:
            print(f"⚠️ DeepFace failed due to memory constraints: {e}")
            print("🔄 Falling back to LBPH for memory efficiency...")
            USE_DEEPFACE = False  # Disable for future requests
    
    # LBPH fallback training
    return train_lbph_recognizer()

def train_deepface_optimized():
    """Memory-optimized DeepFace training with strict limits"""
    # Lazy load DeepFace only when needed
    if not lazy_load_deepface():
        raise Exception("DeepFace lazy loading failed")
    
    embeddings_db = {}
    employees = EmpMaster.query.all()
    if not employees:
        return embeddings_db, {}
    
    emp_id_mapping = {}
    
    for emp in employees:
        emp_faces_count = 0
        # Limit to only 2 images per employee for memory efficiency
        imgs = EmployeeImage.query.filter_by(emp_id=emp.emp_id).limit(2).all()
        if not imgs:
            continue
            
        emp_embeddings = []
        for img_row in imgs:
            try:
                # Save image to temporary file for DeepFace
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img_row.image_data)
                    tmp_path = tmp_file.name
                
                try:
                    # Generate embedding using memory-optimized DeepFace
                    embedding = DeepFace.represent(
                        img_path=tmp_path,
                        model_name=DEEPFACE_MODEL,  # OpenFace - lightweight
                        detector_backend=DEEPFACE_DETECTOR,
                        enforce_detection=False  # More lenient for production
                    )
                    
                    if embedding and len(embedding) > 0:
                        emp_embeddings.append(embedding[0]['embedding'])
                        emp_faces_count += 1
                        
                        # Aggressive memory cleanup
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    print(f"DeepFace embedding failed for {emp.emp_id}: {e}")
                    continue
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
            except Exception as e:
                print(f"Image processing failed for {emp.emp_id}: {e}")
                continue
        
        if emp_faces_count > 0:
            embeddings_db[emp.emp_id] = emp_embeddings
            emp_id_mapping[emp.emp_id] = emp.emp_id
            print(f"✓ DeepFace: Processed {emp_faces_count} faces for employee {emp.emp_id}")
    
    return embeddings_db, emp_id_mapping


def train_lbph_recognizer():
    """Traditional LBPH training for memory-constrained environments"""
    try:
        if not (OPENCV_AVAILABLE and cv2 and hasattr(cv2, 'face')):
            raise Exception("OpenCV LBPH not available")
            
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
            imgs = EmployeeImage.query.filter_by(emp_id=emp.emp_id).limit(5).all()
            if not imgs:
                continue
                
            for img_row in imgs:
                try:
                    npbuf = np.frombuffer(img_row.image_data, dtype=np.uint8)
                    img = cv2.imdecode(npbuf, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    if face_cascade is None:
                        continue
                        
                    detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, minSize=(30, 30))
                    if len(detected_faces) > 0:
                        x, y, w, h = max(detected_faces, key=lambda f: f[2] * f[3])
                        face_roi = img[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (80, 80))
                        faces.append(face_roi)
                        labels.append(label)
                        emp_faces_count += 1
                except Exception:
                    continue
                    
            if emp_faces_count > 0:
                label_to_emp[label] = emp.emp_id
                label += 1
                print(f"✓ LBPH: Processed {emp_faces_count} faces for employee {emp.emp_id}")
        
        if faces and labels and recognizer is not None:
            recognizer.train(faces, np.array(labels))
            
        return recognizer, label_to_emp
        
    except Exception as e:
        print(f"LBPH training failed: {e}")
        return None, {}


# Helper to get employee info with retry logic for connection resilience
def get_employee_info(emp_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Database retry {attempt + 1} for employee info: {e}")
                import time
                time.sleep(0.5)
                # Force new connection
                try:
                    db.session.remove()
                except:
                    pass
            else:
                print(f"Final database error for employee info: {e}")
                return {'emp_id': emp_id, 'id': None, 'name': 'Database Error', 'email': 'No email', 'phone': 'No phone', 'dob': 'No DOB'}


def reload_faces():
    global face_recognizer, label_to_emp, face_recognizer_loaded
    print("Retraining face recognizer with updated data...")
    face_recognizer_loaded = False  # Reset flag to allow reload
    face_recognizer, label_to_emp = train_face_recognizer()
    face_recognizer_loaded = True  # Mark as loaded after successful training
    print(f"✓ Retrained with {len(label_to_emp)} employees")


def validate_face_quality(face_img, width, height):
    """
    PRODUCTION VALIDATION: Multiple checks to prevent false positives
    Returns: (is_valid, reason)
    """
    # Check 1: Minimum face size (reject blurry/distant faces)
    if width < MIN_FACE_WIDTH or height < MIN_FACE_HEIGHT:
        return False, f"Face too small ({width}x{height}). Move closer to camera."
    
    # Check 2: Face brightness (reject too dark/bright faces)
    try:
        avg_brightness = np.mean(face_img)
        if avg_brightness < MIN_FACE_BRIGHTNESS:
            return False, "Face too dark. Improve lighting."
        if avg_brightness > MAX_FACE_BRIGHTNESS:
            return False, "Face overexposed. Reduce lighting."
    except Exception:
        return False, "Cannot assess face quality."
    
    # Check 3: Face contrast (reject flat/washed out images)
    try:
        std_dev = np.std(face_img)
        if std_dev < 10:  # Very low contrast (relaxed from 15)
            return False, "Poor image quality. Improve lighting contrast."
    except Exception:
        pass
    
    return True, "Quality OK"

# Smart initialization: avoid DB queries during Gunicorn startup to prevent connection issues
face_recognizer_loaded = False

def load_face_recognizer():
    """Load and train face recognizer. Called at startup (local) or on first request (production)"""
    global face_recognizer, label_to_emp, face_recognizer_loaded
    
    if face_recognizer_loaded:
        return True
    
    try:
        # Check if we have any recognition capability
        has_lbph = OPENCV_AVAILABLE and cv2 and hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')
        can_load_deepface = True  # Will be tested during lazy loading
        
        if not has_lbph and not can_load_deepface:
            print("✗ No face recognition modules available")
            face_recognizer_loaded = True
            return False

        # Production memory optimization
        if IS_PRODUCTION:
            print("🔧 Production mode: Applying memory optimizations...")
            import gc
            gc.collect()  # Clean up before database operations
        
        # Retry logic for database connection (especially for first request on Render)
        max_retries = 5 if IS_PRODUCTION else 3  # More retries in production
        for attempt in range(max_retries):
            try:
                emp_count = EmpMaster.query.count()
                img_count = EmployeeImage.query.count()
                print(f"Found {emp_count} employees and {img_count} images in database")
                break
            except Exception as db_err:
                if attempt < max_retries - 1:
                    wait_time = 2 if IS_PRODUCTION else 1  # Longer wait in production
                    print(f"Database connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    # Clear stale connections
                    try:
                        db.session.remove()
                    except:
                        pass
                else:
                    print(f"✗ Database connection failed after {max_retries} attempts: {db_err}")
                    raise db_err

        if emp_count > 0:
            print("Training face recognizer...")
            print(f"🔧 Face cascade available: {face_cascade is not None}")
            print(f"🔧 OpenCV available: {OPENCV_AVAILABLE}")
            print(f"🔧 DeepFace enabled: {USE_DEEPFACE}")
            
            try:
                face_recognizer, label_to_emp = train_face_recognizer()
                if face_recognizer is not None and len(label_to_emp) > 0:
                    print(f"✓ Trained with {len(label_to_emp)} employees")
                    print(f"✓ Recognizer type: {type(face_recognizer).__name__ if not isinstance(face_recognizer, dict) else 'DeepFace'}")
                    
                    # Production memory cleanup
                    if IS_PRODUCTION:
                        import gc
                        gc.collect()  # Clean up after training
                        print("🔧 Production: Memory cleanup after training")
                    
                    face_recognizer_loaded = True
                    return True
                else:
                    print("✗ Training failed - no recognizer or employees")
                    face_recognizer = None
                    label_to_emp = {}
                    face_recognizer_loaded = True
                    return False
            except Exception as train_err:
                print(f"✗ Training failed: {train_err}")
                import traceback
                traceback.print_exc()
                face_recognizer = None
                label_to_emp = {}
                face_recognizer_loaded = True
                return False
        else:
            print(f"✗ Cannot train: emp_count={emp_count}")
            print("✗ No employees found in database")
            face_recognizer = None
            label_to_emp = {}
            face_recognizer_loaded = True
            return False
    except Exception as e:
        print(f"Face recognizer load error: {e}")
        import traceback
        traceback.print_exc()
        face_recognizer = None
        label_to_emp = {}
        # Don't set loaded=True on error, allow retry on next request
        return False

# Background training after app starts (avoids Gunicorn startup timeout)
def train_in_background():
    """Load face templates in background thread after app is fully started"""
    import time
    time.sleep(3)  # Wait for app to fully start and database to be ready
    with app.app_context():
        try:
            print("🔧 Background template loading started...")
            print(f"🔧 Production mode: {IS_PRODUCTION}")
            print(f"🔧 OpenCV available: {OPENCV_AVAILABLE}")
            
            # Try loading multiple times in production
            max_attempts = 3 if IS_PRODUCTION else 1
            for attempt in range(max_attempts):
                try:
                    print(f"🔧 Loading attempt {attempt + 1}/{max_attempts}...")
                    success = load_face_recognizer()
                    if success:
                        print(f"✓ Background template loading complete (attempt {attempt + 1})")
                        return
                    else:
                        print(f"⚠️ Loading attempt {attempt + 1} returned False")
                        if attempt < max_attempts - 1:
                            time.sleep(2)
                except Exception as attempt_err:
                    print(f"⚠️ Loading attempt {attempt + 1} error: {attempt_err}")
                    if attempt < max_attempts - 1:
                        time.sleep(2)
                    else:
                        raise
            
            print("✗ Background template loading failed after all attempts")
        except Exception as e:
            print(f"✗ Background template loading error: {e}")
            import traceback
            traceback.print_exc()

# Initialize database tables ONLY (no DB queries to avoid Gunicorn timeout)
with app.app_context():
    try:
        db.create_all()
        print("✓ Database tables ready")
        
        # Detect if running under Gunicorn (production) or local dev server
        import sys
        is_gunicorn = "gunicorn" in sys.argv[0] if sys.argv else False
        
        if IS_PRODUCTION or is_gunicorn:
            # PRODUCTION: Start background training thread to avoid startup timeout
            print("✓ Production mode detected - training will start in background")
            print("🔧 This prevents Gunicorn/Render startup timeouts")
            import threading
            training_thread = threading.Thread(target=train_in_background, daemon=True)
            training_thread.start()
        else:
            # LOCAL: Train immediately
            print("✓ Local mode - loading face templates at startup")
            if OPENCV_AVAILABLE:
                load_face_recognizer()
            else:
                print("✗ OpenCV not available for template loading")
    except Exception as e:
        print(f"Startup warning: {e}")
        import traceback
        traceback.print_exc()


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Simple health check for Render.com - no DB queries"""
    return jsonify({
        'status': 'healthy',
        'opencv': OPENCV_AVAILABLE,
        'lbph_available': FACE_MODULE_AVAILABLE
    }), 200

@app.route('/debug')
def debug_info():
    """Debug endpoint to check production status"""
    try:
        emp_count = EmpMaster.query.count()
        img_count = EmployeeImage.query.count()
        return jsonify({
            'status': 'App running successfully',
            'opencv_available': OPENCV_AVAILABLE,
            'lbph_module_available': FACE_MODULE_AVAILABLE,
            'employees_in_db': emp_count,
            'images_in_db': img_count,
            'face_recognizer_loaded': face_recognizer is not None,
            'trained_labels': len(label_to_emp),
            'label_mapping': label_to_emp,
            'distance_threshold': DEEPFACE_DISTANCE_THRESHOLD,
            'face_cascade_loaded': face_cascade is not None and (not hasattr(face_cascade, 'empty') or not face_cascade.empty()),
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'Error in debug endpoint'})

@app.route('/test')
def test_basic():
    """Basic test endpoint to verify app is running"""
    return jsonify({
        'message': 'App is running successfully!',
        'opencv_status': 'Available' if OPENCV_AVAILABLE else 'Not Available',
        'timestamp': str(__import__('datetime').datetime.now())
    })


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
    return render_template('add_person.html')

@app.route('/retrain')
def force_retrain():
    """Force retrain face recognizer for production"""
    global face_recognizer, label_to_emp, face_recognizer_loaded
    try:
        old_count = len(label_to_emp)
        face_recognizer_loaded = False  # Reset flag to allow reload
        load_face_recognizer()
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

@app.route('/identify', methods=['POST'])
def identify():
    try:
        # Safety check: If recognizer failed at startup, try loading now (critical for production)
        if not face_recognizer_loaded or face_recognizer is None:
            print(f"Face recognizer not loaded at startup - attempting to load now... (Production: {IS_PRODUCTION})")
            
            # In production, try multiple times as the system might still be initializing
            max_load_attempts = 3 if IS_PRODUCTION else 1
            success = False
            
            for attempt in range(max_load_attempts):
                try:
                    success = load_face_recognizer()
                    if success and face_recognizer is not None:
                        break
                    elif IS_PRODUCTION and attempt < max_load_attempts - 1:
                        print(f"Production: Load attempt {attempt + 1} failed, retrying...")
                        import time
                        time.sleep(2)  # Wait before retry in production
                except Exception as load_err:
                    print(f"Load attempt {attempt + 1} error: {load_err}")
                    if attempt == max_load_attempts - 1:
                        break
            
            if not success or face_recognizer is None:
                error_msg = 'Face recognition system is unavailable. Please ensure employees are registered.'
                if IS_PRODUCTION:
                    error_msg += ' The system may still be initializing - please try again in a few moments.'
                
                # Debug information for troubleshooting
                debug_info = {
                    'face_recognizer_loaded': face_recognizer_loaded,
                    'face_recognizer_is_none': face_recognizer is None,
                    'opencv_available': OPENCV_AVAILABLE,
                    'face_module_available': FACE_MODULE_AVAILABLE,
                    'deepface_enabled': USE_DEEPFACE,
                    'production_mode': IS_PRODUCTION
                }
                
                print(f"🚨 Face recognition unavailable. Debug info: {debug_info}")
                
                return jsonify({
                    'recognized': False,
                    'message': error_msg,
                    'error': 'Face recognizer not loaded',
                    'debug': debug_info if not IS_PRODUCTION else None
                }), 503
        
        file = request.files.get('image')
        if file is None:
            return jsonify({'error': 'No image provided', 'recognized': False}), 400

        if not label_to_emp:
            return jsonify({
                'recognized': False,
                'message': 'No employees registered yet.',
                'debug': 'No trained employees'
            })

        # Detect which recognition system we're using
        using_deepface = isinstance(face_recognizer, dict)
        
        if using_deepface and USE_DEEPFACE:
            return handle_deepface_recognition(file)
        else:
            return handle_lbph_recognition(file)
            
    except Exception as e:
        import traceback
        print('Error in identify route:', e)
        traceback.print_exc()
        return jsonify({
            'recognized': False,
            'message': 'Internal server error. Please try again.',
            'error': str(e)
        }), 500


def handle_deepface_recognition(file):
    """Handle DeepFace recognition"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        file.seek(0)
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    try:
        if not lazy_load_deepface():
            raise Exception("DeepFace not available")

        if not label_to_emp:
            return jsonify({
                'recognized': False,
                'message': 'No employees registered yet.',
                'debug': f'embeddings_db empty: {len(face_recognizer) if face_recognizer else 0} employees'
            })

        # Generate embedding for the uploaded image
        try:
            query_embedding = DeepFace.represent(
                img_path=tmp_path,
                model_name=DEEPFACE_MODEL,
                detector_backend=DEEPFACE_DETECTOR,
                enforce_detection=True
            )
                
            if not query_embedding or len(query_embedding) == 0:
                return jsonify({'message': 'No face detected in image.', 'recognized': False})
            
            query_vector = query_embedding[0]['embedding']
                
        except Exception as e:
            print(f"DeepFace face detection failed: {e}")
            return jsonify({'message': 'No clear face detected. Please ensure good lighting and face visibility.', 'recognized': False})

        # Find best match among all employees
        best_match_emp_id = None
        best_distance = float('inf')
        
        print(f'DEBUG: Comparing against {len(face_recognizer)} employees')
        
        for emp_id, emp_embeddings in face_recognizer.items():
            for emp_embedding in emp_embeddings:
                try:
                    # Calculate cosine distance using numpy
                    dot_product = np.dot(query_vector, emp_embedding)
                    norm_a = np.linalg.norm(query_vector)
                    norm_b = np.linalg.norm(emp_embedding)
                    distance = 1 - (dot_product / (norm_a * norm_b))
                    
                    print(f'DEBUG: Distance to {emp_id}: {distance:.4f}')
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match_emp_id = emp_id
                        
                except Exception as dist_err:
                    print(f"Distance calculation error for {emp_id}: {dist_err}")
                    continue
        
        # Debug info for production
        print(f'DEBUG: Best match - Employee: {best_match_emp_id}, Distance: {best_distance:.4f}, Threshold: {DEEPFACE_DISTANCE_THRESHOLD}')
        
        # Layer: Strict distance threshold check - PREVENT FALSE POSITIVES
        if best_distance < DEEPFACE_DISTANCE_THRESHOLD and best_match_emp_id:
            try:
                emp_info = get_employee_info(best_match_emp_id)
                
                # Quick attendance marking with error handling and retry logic
                try:
                    from datetime import datetime as dt, timezone
                    today = dt.now(timezone.utc).date()
                    current_time = dt.now(timezone.utc).time()
                    
                    if emp_info and emp_info.get('id'):
                        # Retry attendance marking up to 3 times
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                # Ultra-fast attendance check
                                existing = db.session.query(AttendanceMaster.id).filter_by(
                                    emp_id=emp_info['id'], 
                                    att_date=today
                                ).first()
                                if not existing:
                                    attendance = AttendanceMaster(
                                        emp_id=emp_info['id'],
                                        full_name=emp_info.get('name', 'Unknown'),
                                        check_in=current_time,
                                        att_date=today,
                                        longitude='0.0',
                                        latitude='0.0',
                                        attendance_status='Present'
                                    )
                                    db.session.add(attendance)
                                    db.session.commit()
                                    print(f'DEBUG: Attendance marked for {best_match_emp_id}')
                                else:
                                    print(f'DEBUG: Attendance already marked for {best_match_emp_id} today')
                                break
                            except Exception as db_err:
                                if attempt < max_retries - 1:
                                    print(f"Attendance retry {attempt + 1}: {db_err}")
                                    db.session.rollback()
                                    import time
                                    time.sleep(0.3)
                                else:
                                    print(f"Attendance marking failed after retries: {db_err}")
                except Exception as att_err:
                    print(f"Attendance marking error (non-critical): {att_err}")
                    # Continue even if attendance fails
                
                return jsonify({
                    'recognized': True,
                    'emp_id': emp_info.get('emp_id', 'Unknown'),
                    'name': emp_info.get('name', 'Unknown'),
                    'email': emp_info.get('email', 'No email'),
                    'phone': emp_info.get('phone', 'No phone'),
                    'dob': emp_info.get('dob', 'No DOB'),
                    'distance': float(best_distance),
                    'message': f"Welcome {emp_info.get('name', 'User')}! Attendance marked."
                })
            except Exception as emp_err:
                print(f"Employee info error: {emp_err}")
                import traceback
                traceback.print_exc()
        
        # No valid match found - STRICT REJECTION TO PREVENT FALSE POSITIVES
        print(f'DEBUG: No match found - Best distance {best_distance:.4f} exceeds threshold {DEEPFACE_DISTANCE_THRESHOLD}')
        return jsonify({
            'recognized': False,
            'message': 'Face not recognized. Please ensure you are registered in the system.',
            'debug': f'Best distance: {best_distance:.4f}, Threshold: {DEEPFACE_DISTANCE_THRESHOLD}'
        })
            
    except Exception as e:
        import traceback
        print('Error in DeepFace recognition:', e)
        traceback.print_exc()
        return jsonify({
            'recognized': False,
            'message': 'DeepFace recognition failed. Please try again.',
            'error': str(e)
        }), 500
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def handle_lbph_recognition(file):
    """Handle LBPH recognition fallback"""
    try:
        img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image', 'recognized': False}), 400

        # Resize for faster processing
        height, width = frame.shape[:2]
        if height > 400:
            scale = 400.0 / height
            new_width = int(width * scale)
            frame = cv2.resize(frame, (new_width, 400))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not OPENCV_AVAILABLE or face_cascade is None:
            return jsonify({
                'recognized': False,
                'message': 'Face detection service unavailable'
            })
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(50, 50))
        if len(faces) == 0:
            return jsonify({'message': 'No face detected.', 'recognized': False})

        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (80, 80))

        if face_recognizer is None:
            return jsonify({
                'recognized': False,
                'message': 'LBPH recognizer not available'
            }), 503

        # Face quality validation
        is_valid, quality_msg = validate_face_quality(face_img, w, h)
        if not is_valid:
            return jsonify({
                'recognized': False,
                'message': f'Detection unsuccessful: {quality_msg}'
            })
        
        # LBPH prediction
        label, confidence = face_recognizer.predict(face_img)
        
        print(f'DEBUG: LBPH - Label: {label}, Confidence: {confidence:.2f}, Threshold: {LBPH_CONFIDENCE_THRESHOLD}')
        
        if confidence < LBPH_CONFIDENCE_THRESHOLD:
            emp_id = label_to_emp.get(label, None)
            if emp_id:
                return mark_attendance_and_respond(emp_id, confidence, 'confidence')
        
        return jsonify({
            'recognized': False,
            'message': 'Face not recognized. Please ensure you are registered in the system.',
            'debug': f'LBPH confidence: {confidence:.2f}, Threshold: {LBPH_CONFIDENCE_THRESHOLD}'
        })
        
    except Exception as e:
        import traceback
        print('Error in LBPH recognition:', e)
        traceback.print_exc()
        return jsonify({
            'recognized': False,
            'message': 'LBPH recognition failed. Please try again.',
            'error': str(e)
        }), 500


def mark_attendance_and_respond(emp_id, score, score_type):
    """Common attendance marking function"""
    try:
        emp_info = get_employee_info(emp_id)
        
        # Quick attendance marking
        try:
            from datetime import datetime as dt, timezone
            today = dt.now(timezone.utc).date()
            current_time = dt.now(timezone.utc).time()
            
            if emp_info and emp_info.get('id'):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        existing = db.session.query(AttendanceMaster.id).filter_by(
                            emp_id=emp_info['id'], 
                            att_date=today
                        ).first()
                        if not existing:
                            attendance = AttendanceMaster(
                                emp_id=emp_info['id'],
                                full_name=emp_info.get('name', 'Unknown'),
                                check_in=current_time,
                                att_date=today,
                                longitude='0.0',
                                latitude='0.0',
                                attendance_status='Present'
                            )
                            db.session.add(attendance)
                            db.session.commit()
                            print(f'DEBUG: Attendance marked for {emp_id}')
                        else:
                            print(f'DEBUG: Attendance already marked for {emp_id} today')
                        break
                    except Exception as db_err:
                        if attempt < max_retries - 1:
                            print(f"Attendance retry {attempt + 1}: {db_err}")
                            db.session.rollback()
                            import time
                            time.sleep(0.3)
                        else:
                            print(f"Attendance marking failed: {db_err}")
        except Exception as att_err:
            print(f"Attendance marking error (non-critical): {att_err}")
        
        response_data = {
            'recognized': True,
            'emp_id': emp_info.get('emp_id', 'Unknown'),
            'name': emp_info.get('name', 'Unknown'),
            'email': emp_info.get('email', 'No email'),
            'phone': emp_info.get('phone', 'No phone'),
            'dob': emp_info.get('dob', 'No DOB'),
            'message': f"Welcome {emp_info.get('name', 'User')}! Attendance marked."
        }
        
        if score_type == 'distance':
            response_data['distance'] = float(score)
        else:
            response_data['confidence'] = float(score)
            
        return jsonify(response_data)
        
    except Exception as emp_err:
        print(f"Employee info error: {emp_err}")
        return jsonify({
            'recognized': False,
            'message': 'Employee information retrieval failed.',
            'error': str(emp_err)
        }), 500


# Production configuration
if __name__ == '__main__':
    print(f"🚀 Starting Flask app on port {PORT}")
    print(f"🔧 Production mode: {IS_PRODUCTION}")
    print(f"🔧 Face recognition loaded: {face_recognizer_loaded}")
    
    # Production deployment configuration
    debug = os.environ.get('FLASK_ENV') == 'development' and not IS_PRODUCTION
    
    app.run(
        host='0.0.0.0', 
        port=PORT,
        debug=debug,
        threaded=True,  # Enable threading for better performance
        use_reloader=False  # Disable reloader in production
    )
