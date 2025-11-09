import serial
import time

GSM_MODULE_SUPPORTED = 0
GSM_MODULE_PHONE_NUM = "+639XXXXXXXXX"
GSM_MODULE_PORT = "COM3"

def alert_system_due_to_collision():
    if GSM_MODULE_SUPPORTED:
        send_gsm_sms_alert(
            port = GSM_MODULE_PORT,
            number = GSM_MODULE_PHONE_NUM,
            message="🚨 Traffic collision detected!"
        )

def send_gsm_sms_alert(
    port="COM3",             # change this to your GSM port, e.g. /dev/ttyUSB0 on Linux
    baud_rate=9600,
    number="+639XXXXXXXXX",  # your phone number
    message="🚨 Collision detected!"
):
    try:
        # Open the serial port
        gsm = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # wait for module to initialize

        # Test communication
        gsm.write(b'AT\r')
        time.sleep(0.5)
        print(gsm.read_all().decode(errors='ignore'))

        # Set text mode
        gsm.write(b'AT+CMGF=1\r')
        time.sleep(0.5)
        print(gsm.read_all().decode(errors='ignore'))

        # Send SMS
        gsm.write(f'AT+CMGS="{number}"\r'.encode())
        time.sleep(0.5)
        gsm.write(message.encode() + b"\x1A")  # Ctrl+Z = 26
        time.sleep(3)
        print("[GSM] SMS sent successfully!")

        gsm.close()

    except Exception as e:
        print(f"[GSM] Failed to send SMS: {e}")

# Example call
send_gsm_sms_alert(
    port="COM3",                # or '/dev/ttyUSB0' on Linux
    number="+639XXXXXXXXX",     # replace with your real number
    message="🚨 Collision detected at intersection!"
)
