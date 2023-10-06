from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
model.train(data="tbdata.yaml", epochs=50)  # train the model
