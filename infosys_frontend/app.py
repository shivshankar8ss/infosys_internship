from flask import Flask, render_template, request, send_file, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads/"
PROCESSED_FOLDER = "processed/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLO model with error handling
try:
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = yolo_net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    classes = open("coco.names").read().strip().split("\n")
except Exception as e:
    yolo_net = None
    out_layers = []
    classes = []
    print(f"Error loading YOLO model: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    operation = request.form["operation"]
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        output_path = os.path.join(PROCESSED_FOLDER, filename)
        
        if operation == "image_processing":
            img = cv2.imread(filepath)
            img_blur = cv2.GaussianBlur(img, (15, 15), 0)
            cv2.imwrite(output_path, img_blur)

        elif operation == "yolo_image" and yolo_net:
            process_yolo_image(filepath, output_path)
        
        elif operation == "yolo_video" and yolo_net:
            process_yolo_video(filepath, output_path)
        
        elif operation == "retinanet_video":
            process_retinanet_video(filepath, output_path)
        
        elif operation == "tracking":
            process_object_tracking(filepath, output_path)

        return send_file(output_path, mimetype="image/png" if "image" in filename else "video/mp4")
    
    return jsonify({"error": "No file uploaded."}), 400

@app.route("/download")
def download():
    files = os.listdir(PROCESSED_FOLDER)
    if files:
        return send_file(os.path.join(PROCESSED_FOLDER, files[0]), as_attachment=True)
    return "No processed file available.", 400

# Function for YOLO image processing
def process_yolo_image(input_path, output_path):
    image = cv2.imread(input_path)
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(out_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, classes[class_ids[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)

# Function for YOLO video processing
def process_yolo_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_yolo_image(frame, frame)
        out.write(frame)
    
    cap.release()
    out.release()

# Placeholder function for RetinaNet video processing
def process_retinanet_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # RetinaNet processing logic here
        out.write(frame)
    
    cap.release()
    out.release()

# Function for object tracking
def process_object_tracking(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    tracker = cv2.TrackerCSRT_create()
    ret, frame = cap.read()
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker.init(frame, bbox)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    app.run(debug=True)
