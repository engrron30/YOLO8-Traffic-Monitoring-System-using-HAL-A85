import cv2
from ultralytics import YOLO


def run_traffic_detection():
    # Load YOLOv8 model (pre-trained on COCO)
    model = YOLO("yolov8n.pt")  # small model; can use yolov8m.pt or yolov8l.pt for more accuracy

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Failed to open stream")
        exit()

    window = "Traffic Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_positions = {}
    frame_id = 0

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

                if label not in ["car", "truck", "bus", "motorbike"]:
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

        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
