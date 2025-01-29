from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('pneumonia_model.h5')  # Load your model

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))  # <-- Change to 224x224
    img = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Shape becomes (1, 224, 224, 3)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)
            result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"
            confidence = round(float(prediction[0][0]) * 100, 2) if result == "Pneumonia Detected" else round((1 - float(prediction[0][0])) * 100, 2)
            
            return render_template('result.html', 
                                 result=result, 
                                 confidence=confidence, 
                                 filename=filename)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)