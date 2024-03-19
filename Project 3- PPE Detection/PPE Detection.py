import math
from ultralytics import YOLO
import cv2
import cvzone
# For Videos
video = cv2.VideoCapture("../Project 3- PPE Detection/ppe-3-1.mp4")
# For webcam
#video = cv2.VideoCapture(0)
video.set(3, 1080)
video.set(4, 720)
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety vest', 'Person', 'Safety Cone',
              'Safety vest', 'machinery', 'vehicle']

model = YOLO('../Project 3- PPE Detection/ppe.pt')
My_color = (0, 250, 0)
while True:
    success, img = video.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            CurrentCLASS = classNames[cls]
            if conf > 0.5:

                if CurrentCLASS == "Hardhat" or CurrentCLASS == "Mask" or CurrentCLASS == "Safety vest":
                    My_color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=My_color, thickness=3)
                    cvzone.putTextRect(img, f"{CurrentCLASS}", (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorR=(0, 255, 0))
                elif CurrentCLASS == "NO-Hardhat" or CurrentCLASS == "NO-Mask" or CurrentCLASS == "NO-Safety vest":
                    My_color = (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=My_color, thickness=3)
                    cvzone.putTextRect(img, f"{CurrentCLASS}", (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorR=(0, 0, 255))
                else:
                    My_color = (255, 0, 0)
                    cvzone.putTextRect(img, f"{CurrentCLASS}", (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, colorR=(255, 0, 0))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=My_color, thickness=3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)