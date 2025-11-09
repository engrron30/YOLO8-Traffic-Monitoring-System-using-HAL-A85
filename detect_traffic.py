import cv2
import time
from ultralytics import YOLO

YOLO_MODEL_USE_TYPE = 1
YOLO_MODEL_DEFAULT = "yolov8n.pt"
YOLO_MODEL_TYPES = [
    (1, "yolov8n.pt"),
    (2, "yolov8m.pt"),
    (3, "yolov8l.pt")
]

DETECT_OBJECTS = [
    "car",
    "truck",
    "bus",
    "motorbike",
]
DETECT_TRAFFIC_COLLISION = 0
DETECT_COLLISION_THRESHOLD_SECONDS = 5
CROSS_LINE_SIZE = 60

def make_model_based_on_conf():
    # Load YOLOv8 model (pre-trained on COCO)
    # small model; can use yolov8m.pt or yolov8l.pt for more accuracy
    model = YOLO_MODEL_DEFAULT

    for model_id, model_name in YOLO_MODEL_TYPES:
        if model_id == YOLO_MODEL_USE_TYPE:
            model = model_name
            break

    return model

"""Check if two bounding boxes overlap"""
def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def run_traffic_detection(camera_url, model_name):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Failed to open stream")
        exit()

    window = "Traffic Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    overlap_timers = {}  # {(id1, id2): start_time}
    frame_id = 0

    if DETECT_TRAFFIC_COLLISION:
        run_traffic_detection_with_collision(cap, model, window, overlap_timers, frame_id)
    else:
        run_traffic_detection_normally(cap, model, window, overlap_timers, frame_id)

def run_traffic_detection_with_collision(cap, model, window, overlap_timers, frame_id):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            break
        frame_id += 1

        results = model(frame, verbose=False)
        boxes_list = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label not in DETECT_OBJECTS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_list.append((label, (x1, y1, x2, y2)))

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Check overlaps for collision
        current_time = time.time()
        new_overlap_timers = {}

        for i in range(len(boxes_list)):
            label1, box1 = boxes_list[i]
            for j in range(i + 1, len(boxes_list)):
                label2, box2 = boxes_list[j]
                if boxes_overlap(box1, box2):
                    key = tuple(sorted([i, j]))
                    start_time = overlap_timers.get(key, current_time)
                    if current_time - start_time >= DETECT_COLLISION_THRESHOLD_SECONDS:
                        # Draw collision alert
                        x1, y1, x2, y2 = box1
                        cx, cy = (x1 + x2)//2, (y1 + y2)//2
                        cv2.putText(frame, "COLLISION!", (cx-50, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                        x1, y1, x2, y2 = box2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)

                    new_overlap_timers[key] = start_time

        overlap_timers = new_overlap_timers
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def run_traffic_detection_normally(cap, model, window, prev_positions, frame_id):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            break
        frame_id += 1

        # Run object detection
        results = model(frame, verbose=False)

        # Draw detections and compute pseudo-speed
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label not in DETECT_OBJECTS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Track approximate motion for pseudo-speed
                prev = prev_positions.get(label, (cx, cy))
                dx = cx - prev[0]
                dy = cy - prev[1]
                pixel_speed = (dx**2 + dy**2) ** 0.5
                prev_positions[label] = (cx, cy)

                # Draw bounding box + speed
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} speed={pixel_speed:.1f}px/f",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                #cv2.line(frame, (cx - CROSS_LINE_SIZE, cy), (cx + CROSS_LINE_SIZE, cy), (255, 255, 224), 2)
                #cv2.line(frame, (cx, cy - CROSS_LINE_SIZE), (cx, cy + CROSS_LINE_SIZE), (255, 255, 224), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # small red dot at the center

        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
