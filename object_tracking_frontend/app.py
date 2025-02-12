from flask import Flask, render_template, request, Response, jsonify
import cv2
import os
from object_detection import ObjectDetection
import math

app = Flask(__name__)
od = ObjectDetection()

if not os.path.exists('output'):
    os.makedirs('output')

cap = None
tracking_objects = {}
track_id = 0

def generate_frames():
    global tracking_objects, track_id, cap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_video_path = os.path.join('output', 'tracked_video.mp4')
    out = None  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        if out is None:
            frame_height, frame_width = frame.shape[:2]
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

       
        center_points_cur_frame = []

        
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box

            
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))

            # Draw rectangle around detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update tracking objects
        for pt in center_points_cur_frame:
            same_object_detected = False
            for object_id, prev_pt in tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])

                if distance < 35:  
                    tracking_objects[object_id] = pt
                    same_object_detected = True
                    break

            # Assign new ID to new object
            if not same_object_detected:
                tracking_objects[track_id] = pt
                track_id += 1

        # Draw tracking points and IDs
        for object_id, pt in tracking_objects.items():
            # Draw a filled circle for tracking
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

           
            cv2.putText(
                frame,
                str(object_id),
                (pt[0] - 10, pt[1] - 10),  
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  
                2, 
                lineType=cv2.LINE_AA,  
            )

        # Write the frame with tracking into the output video
        out.write(frame)

        # Encode frame to jpeg for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Release the VideoWriter and cap after the loop ends
    out.release()
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global cap, tracking_objects, track_id

    # Reset previous tracking data
    tracking_objects = {}
    track_id = 0

    video_file = request.files.get('video') 
    if video_file:
        video_path = os.path.join('output', video_file.filename)
        video_file.save(video_path)

        # Open the uploaded video file for processing
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return jsonify(success=False, message="Failed to open video.")

        return jsonify(success=True, message="Video uploaded and tracking started.")

    return jsonify(success=False, message="No video uploaded.")

@app.route('/download_video', methods=['GET'])
def download_video():
    tracked_video_path = os.path.join('output', 'tracked_video.mp4')
    if os.path.exists(tracked_video_path):
        return jsonify(success=True, video_url=tracked_video_path)
    else:
        return jsonify(success=False, message="Tracked video not found.")

if __name__ == '__main__':
    app.run(debug=True)
