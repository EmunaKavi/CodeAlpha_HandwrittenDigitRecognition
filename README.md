# Handwritten Digit Recognition

A web-based application that recognizes handwritten digits (0-9) using deep learning. This project combines a Convolutional Neural Network (CNN) model with a user-friendly web interface for real-time digit prediction.

## 🎯 Project Overview

This project was developed as part of the CodeAlpha Machine Learning Internship program. It demonstrates the implementation of image processing and deep learning techniques to identify handwritten digits drawn on a canvas.

## ✨ Features

- **Interactive Drawing Canvas**: Draw digits directly in your browser
- **Real-time Prediction**: Get instant predictions using a trained CNN model
- **Clean Interface**: Simple and intuitive user experience
- **REST API**: FastAPI backend for model serving
- **Preprocessing Pipeline**: Automatic image resizing and normalization

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: FastAPI (Python)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: PIL (Python Imaging Library)
- **Model**: Convolutional Neural Network (CNN)

## 📁 Project Structure

```
CodeAlpha_HandwrittenDigitRecognition/
│
├── index.html          # Frontend interface with drawing canvas
├── app.py             # FastAPI backend server
├── model.h5           # Trained CNN model (not included in repo)
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/CodeAlpha_HandwrittenDigitRecognition.git
cd CodeAlpha_HandwrittenDigitRecognition
```

### Step 2: Install Dependencies

```bash
pip install fastapi uvicorn tensorflow pillow numpy python-multipart
```

Or create a `requirements.txt` file:

```txt
fastapi==0.104.1
uvicorn==0.24.0
tensorflow==2.13.0
pillow==10.0.1
numpy==1.24.3
python-multipart==0.0.6
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 3: Train Your Model (Optional)

If you don't have a pre-trained model, you can train one using the MNIST dataset:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save the model
model.save('model.h5')
```

### Step 4: Run the Application

1. Start the FastAPI server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

2. Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## 🎮 How to Use

1. **Draw a Digit**: Use your mouse to draw a digit (0-9) on the black canvas
2. **Clear Canvas**: Click the "Clear" button to erase your drawing
3. **Get Prediction**: Click the "Predict" button to see what digit the model thinks you drew
4. **View Result**: The predicted digit will appear below the buttons

## 🔧 API Endpoints

### GET `/`
Returns the main HTML interface.

### POST `/predict`
Accepts an image file and returns the predicted digit.

**Request**: Multipart form data with image file
**Response**: JSON object with predicted digit
```json
{
  "digit": 7
}
```

## 🧠 Model Architecture

The CNN model uses the following architecture:
- **Input Layer**: 28x28x1 (grayscale images)
- **Convolutional Layers**: Feature extraction with ReLU activation
- **MaxPooling Layers**: Dimensionality reduction
- **Dense Layers**: Classification with softmax output
- **Output**: 10 classes (digits 0-9)

## 📊 Model Performance

The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. Typical performance metrics:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%
- **Model Size**: ~1.2MB

## 🔄 Image Processing Pipeline

1. **Canvas Drawing**: User draws on 280x280 HTML5 canvas
2. **Image Capture**: Canvas content converted to PNG format
3. **Preprocessing**: 
   - Convert to grayscale
   - Resize to 28x28 pixels
   - Normalize pixel values (0-1)
   - Reshape for model input (1, 28, 28, 1)
4. **Prediction**: Model predicts digit class
5. **Result Display**: Predicted digit shown to user

## 🚀 Deployment Options

### Local Development
- Use `uvicorn` for local testing and development

### Production Deployment
- **Heroku**: Deploy using Procfile
- **Docker**: Containerize the application
- **AWS/GCP**: Deploy on cloud platforms
- **Vercel/Netlify**: For frontend deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CodeAlpha**: For providing the internship opportunity and project guidelines
- **MNIST Dataset**: For the handwritten digit dataset
- **TensorFlow Team**: For the deep learning framework
- **FastAPI Team**: For the modern web framework

## 📞 Contact

- **Developer**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

## 🏆 Project Status

This project was completed as part of the CodeAlpha Machine Learning Internship program. It demonstrates proficiency in:
- Deep Learning with TensorFlow
- Web Development with FastAPI
- Image Processing techniques
- Full-stack application development

---

**Note**: Make sure to include your trained `model.h5` file in the project directory before running the application. The model file is not included in the repository due to size constraints.
