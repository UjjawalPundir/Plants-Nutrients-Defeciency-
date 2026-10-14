
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50_plant_nutrient_model.pth"

CLASS_NAMES = ["K", "N", "healthy"]
CONFIDENCE_THRESHOLD = 70.0

device = torch.device("cpu")

# Load your existing trained model
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, len(CLASS_NAMES))

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    error = None
    filename = None
    result_status = None
    message = None

    if request.method == "POST":
        file = request.files.get("plant_image")

        if file is None or file.filename == "":
            error = "Please select a plant image first."

        elif not file.filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            error = "Please upload a JPG, PNG, BMP, or WEBP image."

        else:
            try:
                filename = secure_filename(file.filename)

                image = Image.open(file.stream).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)

                    predicted_index = probabilities.argmax(dim=1).item()
                    confidence = round(
                        probabilities[0, predicted_index].item() * 100,
                        2
                    )

                if confidence < CONFIDENCE_THRESHOLD:
                    result_status = "uncertain"
                    prediction = "Uncertain prediction"
                    message = (
                        "The model is not sufficiently confident. "
                        "Please upload a clear, close-up tomato leaf image."
                    )

                else:
                    result_status = "prediction"
                    prediction = CLASS_NAMES[predicted_index]

                    if prediction == "healthy":
                        message = (
                            "The model predicts a healthy-class leaf. "
                            "This result is not a confirmed diagnosis."
                        )
                    elif prediction == "K":
                        message = (
                            "The model predicts the K deficiency class."
                        )
                    elif prediction == "N":
                        message = (
                            "The model predicts the N deficiency class."
                        )

            except (UnidentifiedImageError, OSError):
                error = (
                    "The file could not be read as an image. "
                    "Please try another image."
                )

            except Exception:
                app.logger.exception("Image analysis failed")
                error = (
                    "Something went wrong while analysing the image."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        filename=filename,
        result_status=result_status,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
