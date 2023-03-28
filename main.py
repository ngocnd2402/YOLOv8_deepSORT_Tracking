import os
import cv2
from ultralytics import YOLO
from tracker import deepSORT_Tracker

video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'people_out.mp4')
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

model = YOLO("yolov8m.pt", task='detect')
tracker = deepSORT_Tracker()
detection_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            track_id, bbox = track
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (102,0,204), 2 )
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)

    cap_out.write(frame)

cap.release()
cap_out.release()
cv2.destroyAllWindows()
