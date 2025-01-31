import os
from flask import Flask, render_template, request, send_from_directory
import torch
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set the path for saving uploaded videos
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video
        output_path = process_video(filepath)

        # Return the processed video
        return send_from_directory(OUTPUT_FOLDER, output_path)

    return 'Invalid file type', 400

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Reduce FPS to slow down the video
    fps = max(fps // 2, 10)  # Ensure FPS does not go below 10

    # Set up video writer to save output video
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame
        results = model(frame)

        # Render the results (bounding boxes and labels) and apply detection confidence
        results.conf = 0.4  # Lower confidence threshold to detect more objects
        frame_with_boxes = results.render()[0]

        # Write the frame to the output video
        out.write(frame_with_boxes)

    cap.release()
    out.release()

    return 'output_video.mp4'

if __name__ == '__main__':
    app.run(debug=True)
