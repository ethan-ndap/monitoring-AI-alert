#  Practical AI-based model for patient monitoring systems

## Introduction and Usage Instructions

This project aims to implement a practical AI-driven model for detecting emergency situations and generating health alerts within a home monitoring system. The system will automatically generate and send alerts to health professionals if it detects any unusual behaviour within a given environment, enabling rapid response to emergency situations. Such a system will have widespread benefits, for example, monitoring elderly individuals in their own home for potential emergencies. Below are the steps to use an existing `.pth` model or train a new one using YOLOv8, and how to send alerts via Twilio.

### Using an Existing `.pth` Model
1. Place your `.pth` model file in the designated `models/` directory.
2. Update the configuration file to point to the `.pth` model path.
3. Run the monitoring script:
  ```shellscript
  python monitor.py --model-path models/your_model.pth
  ```

### Training a New Model with YOLOv8
1. Install YOLOv8 dependencies:
  ```shellscript
  pip install ultralytics
  ```
2. Train the model:
  ```shellscript
  yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
  ```
3. Export the trained model:
  ```shellscript
  yolo export model=best.pt format=torchscript
  ```
4. Move the exported model to the `models/` directory.

### Sending Alerts via Twilio
1. Install the Twilio Python SDK:
  ```shellscript
  pip install twilio
  ```
2. Update the `config.py` file with your Twilio credentials (`account_sid`, `auth_token`, and `from_phone_number`).
3. Start your Node.js server:
  ```shellscript
  node server.js
  ```
4. Run the development script:
  ```shellscript
  npm run dev
  ```
5. Use the following script to send a message:
  ```shellscript
  python send_alert.py --to-phone-number +1234567890 --message "Alert: Patient condition requires attention!"
  ```

Developing a practical AI-based model for patient monitoring systems
