from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import sqlite3
import os
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
from tensorflow.keras.utils import get_custom_objects
from trainmodel.train_model import CustomCNN
from tensorflow.keras.utils import get_custom_objects
import numpy as np

app = Flask(__name__)
app.secret_key = '01lo'
get_custom_objects().update({'ICustomCNN': CustomCNN})

model = load_model("custom_plant_disease_model.keras")

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Helper: Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database connection
def connect_db():
    conn = sqlite3.connect('database.db')
    return conn
#for showing classification chart
import matplotlib.pyplot as plt

def generate_classification_chart(stats):
    labels, counts = zip(*stats) if stats else ([], [])
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color='green')
    plt.xlabel('Disease')
    plt.ylabel('Number of Classifications')
    plt.title('Classification Trends')
    plt.tight_layout()
    chart_path = 'static/charts/classification_chart.png'
    plt.savefig(chart_path)
    return chart_path


# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['image']
    farmer_name = request.form['farmer_name']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict using the model
        try:
            image = load_img(file_path, target_size=(256, 256))  
            images = img_to_array(image) / 255.0
            images = np.expand_dims(images, axis=0)
            print(images)
            predictions = model.predict(images)
            print(f"Predictions (raw probabilities): {predictions}")
            
            labels = ["Early Blight", "Potato___Late_blight", "Potato___healthy"]
             # Convert predictions to a dictionary of label-confidence pairs
            prediction_dict = {labels[i]: predictions[0][i] for i in range(len(labels))}
            result = max(prediction_dict, key=prediction_dict.get)
            confidence = round(prediction_dict[result] * 100, 2)
            
            print(f"Predicted Label: {result}")
            print(f"Confidence: {confidence}%")
            for label, prob in prediction_dict.items():
                print(f"{label}: {round(prob * 100, 2)}%")
            
            
            # Save to database
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO uploads (farmer_name, image_path, result,confidence) VALUES (?, ?, ?,?)", 
                           (farmer_name, filename, result,confidence))
            conn.commit()
            conn.close()

            return render_template('result.html', farmer_name=farmer_name, result=result, confidence=confidence, filename=filename)

        except Exception as e:
            flash(f"Error during prediction: {e}")
            return redirect(url_for('home'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin WHERE username = ? AND password = ?", (username, hashed_password))
        admin = cursor.fetchone()
        conn.close()
        if admin:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Query statistics
    cursor.execute("SELECT COUNT(*) FROM uploads")
    total_uploads = cursor.fetchone()[0]
    
    cursor.execute("SELECT result, COUNT(result) FROM uploads GROUP BY result")
    classification_stats = cursor.fetchall()
    
    cursor.execute("SELECT * FROM uploads")
    data = cursor.fetchall()

     # Fetch feedback messages
    cursor.execute("SELECT farmer_name, message FROM feedback")
    feedback_data = cursor.fetchall()
    
    conn.close()
    
    # Graph generation (optional)
    chart_path = generate_classification_chart(classification_stats)
    
    return render_template(
        'dashboard.html', 
        data=data, 
        total_uploads=total_uploads, 
        classification_stats=classification_stats, 
        chart_path=chart_path,
        feedback=feedback_data
    )

#feedback

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        farmer_name = request.form['farmer_name']
        feedback_message = request.form['feedback_message']
        
        # Save feedback in the database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (farmer_name, message) VALUES (?, ?)", 
                       (farmer_name, feedback_message))
        conn.commit()
        conn.close()
        
        flash("Thank you for your feedback!")
        return redirect(url_for('feedback_success'))
    return render_template('feedback.html')


@app.route('/feedback_success')
def feedback_success():
    return render_template('feedback_success.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)