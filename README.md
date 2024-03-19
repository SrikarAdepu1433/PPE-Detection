# Personal Protective Equipment (PPE) Detection in Construction Fields

This repository contains Python code for detecting personal protective equipment (PPE) in construction fields using the YOLO (You Only Look Once) object detection model. The system identifies various types of PPE such as hardhats, masks, and safety vests in a video stream or webcam feed.

## Dependencies
- `math`
- `cv2` (OpenCV)
- `cvzone`
- `ultralytics` (for YOLO object detection)

## Usage
1. Clone the repository.
2. Install the dependencies using pip.
3. Download the YOLO weights (e.g., `ppe.pt`) and place them in the specified directory.
4. Run the Python script (`ppe_detection.py`) with a video file or webcam feed as input.
5. View the output with PPE detection results displayed on the screen.

## Description
- The code utilizes the YOLO object detection model trained on PPE classes to detect objects in each frame of the video.
- Detected objects are classified into PPE categories such as hardhats, masks, and safety vests based on predefined class labels.
- Detected PPE items are highlighted with bounding boxes and labeled with their corresponding classes.
- Different colors are used for visualization based on the type of PPE detected (e.g., green for positive detections, red for negative detections).
- The code can be easily customized to adjust the confidence threshold for detection or add new PPE classes as needed.

## Note
- Ensure that the YOLO model weights (`ppe.pt`) are available in the specified directory for proper functioning of the code.
- Adjustments to parameters such as confidence thresholds or color schemes can be made in the code according to specific requirements.
- This system can serve as a tool for enhancing safety measures in construction sites by automatically monitoring PPE compliance among workers.

