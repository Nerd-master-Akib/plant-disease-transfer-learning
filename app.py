
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.model import PlantResNet18


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Classes
# -----------------------------

classes = [
    "Potato Early Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]


# -----------------------------
# Load Model
# -----------------------------

model = PlantResNet18().to(device)

model.load_state_dict(
    torch.load(
        "plant_resnet18.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# Image Transformation
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload an image of a potato or tomato leaf "
    "and the AI model will predict its condition."
)


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image"
    )

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = classes[predicted.item()]

    confidence_percentage = confidence.item() * 100



    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence_percentage:.2f}%"
    )

