# 🌱 Tomato Nutrient Deficiency Detection Using Deep Learning

## 📌 Project Overview

Plants Nutrient Deficiency Detection is a deep learning-based web application designed to analyze tomato leaf images and identify potential nutrient deficiencies.

The application uses a **ResNet50 convolutional neural network** trained with PyTorch and a Flask-based web interface that allows users to upload leaf images and view the predicted class with its confidence score.

The current model supports three classes:
- **K** — Potassium deficiency
- **N** — Nitrogen deficiency
- **Healthy** — Healthy leaf

**Note:** The current nutrient classification model has only these three classes. It does not independently verify whether an uploaded image belongs to a tomato plant. Predictions for other plant species may be incorrect.

## ✨ Features

- Upload leaf images through a web interface.
- Predict supported nutrient deficiency categories.
- Display prediction confidence.
- Use a ResNet50 deep learning model.
- Run predictions through a Flask application.
- Simple and user-friendly interface.

## 🛠️ Technologies Used

- Python
- Flask
- PyTorch
- Torchvision
- ResNet50
- Pillow
- HTML and CSS

## 📂 Project Structure

```text
Plants-Nutrients-Defeciency-/
│
├── app.py
├── requirements.txt
├── .gitignore
├── resnet50_plant_nutrient_model.pth
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

*The structure above is an example. Adjust the filenames to match your actual project.*

## 📥 Download the Trained Model

The trained model file is hosted separately because it exceeds GitHub's standard browser-upload limit.

**Google Drive Model Link:**  
https://drive.google.com/file/d/1dynOn_gitLmzkxfv2kS-V3VBMhXz9Aqb/view?usp=drive_link

### Model Setup

1. Open the Google Drive link above.
2. Download `resnet50_plant_nutrient_model.pth`.
3. Place the downloaded file in the same directory as `app.py`.

The application expects the model at:

```text
resnet50_plant_nutrient_model.pth
```

## ⚙️ Installation and Setup

### 1. Install Python

Install a compatible version of Python on your computer.

### 2. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/UjjawalPundir/Plants-Nutrients-Defeciency-.git
```

Move into the project directory:

```bash
cd Plants-Nutrients-Defeciency-
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install a compatible PyTorch and Torchvision version for your system.

### 5. Download the Model

Download the trained `.pth` file from the Google Drive link and save it in the project root directory, next to `app.py`.

### 6. Run the Application

```bash
python app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://127.0.0.1:5000
```

## 🖥️ How to Use

1. Open the web application in your browser.
2. Select a tomato leaf image.
3. Submit the image for analysis.
4. View the predicted nutrient category and confidence score.

## ⚠️ Limitations

- The model currently recognizes only the classes on which it was trained.
- The current model does not guarantee that an image belongs to a tomato plant.
- Prediction confidence is not a guarantee of correctness.
- Results depend on image quality and the training dataset.
- The model is intended for educational and experimental use, not as a substitute for laboratory nutrient analysis.

## 🚀 Future Improvements

- Add a tomato-versus-other-plant classifier.
- Expand the dataset to include additional nutrient deficiencies.
- Improve accuracy using more diverse training images.
- Add prediction history and image previews.
- Deploy the application to a public hosting platform.

## 👨‍💻 Author

**Ujjawal Pundir**

GitHub: [@UjjawalPundir](https://github.com/UjjawalPundir)

## 📄 License

Add a suitable open-source license if you intend to allow others to reuse or modify this project.
