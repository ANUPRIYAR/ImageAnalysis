import cv2
from PIL import Image

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
# model = Model()
# model._load(r"C:\Users\ANRAMAC\PycharmProjects\InstanceSegmentationUnderwritting\yolo11n-seg.pt", task="detect")
# print(model)
# # model._load("yolo11n.pt")
# # model = _load(weights: Union[str, Path] = 'yolo11n.pt')
image_path = r"C:\Users\ANRAMAC\Downloads\OneDrive_2025-01-09\Sample Images\Watch2.jpg"
# Run batched inference on a list of images
results = model([image_path])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
