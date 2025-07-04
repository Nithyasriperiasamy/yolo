import cv2
import time
import os
from ultralytics import YOLO
from gtts import gTTS
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import queue
import threading

# Email details
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_email_password"
RECIPIENT_EMAIL = "recipient_email@example.com"

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Audio Alert Functions
def play_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("buzzer.mp3")
    pygame.mixer.music.play()

def say_person_detected():
    tts = gTTS(text="Person detected", lang='en')
    audio_file = "Buzzer.wav"
    tts.save(audio_file)
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(audio_file)

# Email Alert Function
def send_email():
    msg = MIMEMultipart()
    msg['From'] = pp8000774@gmail.com
    msg['To'] = nithyasriperiyasamy@gmail.com
    msg['Subject'] = "Alert: Person Detected!"
    body = "A person has been detected by the YOLOv8 security system."
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
    except Exception as e:
        print("Failed to send email:", str(e))

# Main Detection Loop
person_detected = False
last_detection_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Class ID 0 corresponds to "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if not person_detected or (time.time() - last_detection_time > 30):
                    play_alarm()
                    say_person_detected()
                    send_email()
                    person_detected = True
                    last_detection_time = time.time()

    cv2.imshow("YOLOv8 Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
