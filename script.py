import cv2
from ultralytics import YOLO

DEMO_LOCAL_STREAM = True            # Set this true if to detect local video for testing
DEMO_ERROR_VIDEO = False            # Set this true if traffic with collision is to test
REMOTE_STREAM_USE_RTSP = True

# Local Demo Files Defines
demo_vid_file_type = "mp4"
demo_vid_dir = "Sample Data"
if DEMO_ERROR_VIDEO:
    demo_vid_name = f"vid-with-malicious-traffic.{demo_vid_file_type}"
else:
    demo_vid_name = f"vid-with-normal-traffic.{demo_vid_file_type}"

# Remote Live Stream Defines
user_name = "hwjk"
user_pass = "pa6tb7"
ipv4_addr = "192.168.1.10"
resource_path = "cam/realmonitor"
channel_num = 1
subtype_num = 0
if REMOTE_STREAM_USE_RTSP:
    remote_protocol = "rtsp"
    remote_port = 554
else:
    remote_protocol = "http"
    remote_port = 80

if DEMO_LOCAL_STREAM:
    url = f"{demo_vid_dir}/{demo_vid_file_type}"
else:
    url = f"{remote_protocol}://{user_name}:{user_pass}@{ipv4_addr}:{remote_port}/{resource_path}?channel={channel_num}&subtype={subtype_num}"

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
