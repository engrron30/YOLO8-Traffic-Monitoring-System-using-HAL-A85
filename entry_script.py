from detect_traffic import run_traffic_detection, make_model_based_on_conf

REMOTE_STREAM_USE_RTSP      = 1
REMOTE_STREAM_IPV4_ADDRESS  = "192.168.1.10"
REMOTE_STREAM_USER_NAME     = "hwjk"
REMOTE_STREAM_USER_PASSWORD = "pa6tb7"

DEMO_LOCAL_STREAM           = 1  # Set this true if to detect local video for testing
DEMO_ERROR_VIDEO            = 1     # Set this true if traffic with collision is to test
DEMO_VID_FILE_TYPE          = "mp4"
DEMO_VID_DIR                = "Sample Data"


def traffic_detection_welcoming_notes():
    print("#################################################")
    print("#################################################")
    print("###                                           ###")
    print("###           TRAFFIC MONITORING              ###")
    print("###         SYSTEM with AI-DETECTION          ###")
    print("###                                           ###")
    print("#################################################")
    print("#################################################\n")

# Prepare configuration before running the actual live stream with
#   AI Detection to define which file or stream to use.
def make_camsource_based_on_conf():
    if DEMO_ERROR_VIDEO:
        demo_vid_name = f"vid-with-malicious-traffic.{DEMO_VID_FILE_TYPE}"
    else:
        demo_vid_name = f"vid-with-normal-traffic.{DEMO_VID_FILE_TYPE}"

    # Remote Live Stream Defines
    user_name = REMOTE_STREAM_USER_NAME
    user_pass = REMOTE_STREAM_USER_PASSWORD
    ipv4_addr = REMOTE_STREAM_IPV4_ADDRESS
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
        conf_url = f"{DEMO_VID_DIR}/{demo_vid_name}"
    else:
        conf_url = f"{remote_protocol}://{user_name}:{user_pass}@{ipv4_addr}:   {remote_port}/{resource_path}?channel={channel_num}&subtype={subtype_num}"

    return conf_url


if __name__ == "__main__":
    traffic_detection_welcoming_notes()

    camera_source = make_camsource_based_on_conf()
    model_name = make_model_based_on_conf()
    run_traffic_detection(camera_source, model_name)