import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image

# Open the camera
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Wrap the model in DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

    if not ret:
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL image
    pil_image = Image.fromarray(np.uint8(rgb_frame)).convert('RGB')

    # Preprocess the image for the model
    inputs = processor(images=pil_image, return_tensors="pt")

    # Forward pass with the model
    with torch.no_grad():
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Move predicted_depth to CPU before interpolating
        predicted_depth = predicted_depth.cpu()

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_frame.shape[:2][::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Move prediction to CPU before visualizing
        prediction = prediction.cpu()

    # Visualize the prediction
    formatted = (prediction * 255 / torch.max(prediction)).to(torch.uint8)
    depth = Image.fromarray(formatted.cpu().numpy())

    # Display the resulting frame and depth map
    cv2.imshow('frame', frame)
    cv2.imshow('depth', np.array(depth))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

