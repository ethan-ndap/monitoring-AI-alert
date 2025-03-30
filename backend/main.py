import csv

import cv2
import json
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
import time
from model import load_trained_model  # Import LSTM model loader

# Load the trained LSTM model
model_lstm = load_trained_model("../elderly_movement_predictor.pth")

# Initialize YOLO for detection
model_yolo = YOLO("yolov8x.pt").to('cuda')

# Define region mapping
REGION_MAPPING = {
    1: "Living Room",
    2: "Kitchen",
    3: "Bedroom"
}

REGION_FILE = "saved_regions.json"

def extract_roi_from_webcam(region_names):
    """ Extracts user-defined ROIs from webcam and saves them to a file. """
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(region_name, frame_copy)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return []

    ROIs = []
    print(f"Using webcam to extract {len(region_names)} regions of interest")

    for region_name in region_names:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from webcam")
            break

        frame_copy = frame.copy()
        cv2.namedWindow(region_name)
        cv2.setMouseCallback(region_name, mouse_callback)

        points = []
        while True:
            cv2.imshow(region_name, frame_copy)
            key = cv2.waitKey(1)
            if key == 27 or len(points) == 4:  # Press ESC or define 4 points
                break

        if len(points) == 4:
            x_range = [min(p[0] for p in points), max(p[0] for p in points)]
            y_range = [min(p[1] for p in points), max(p[1] for p in points)]
            region = {"name": region_name, "polygon": points, "range": [x_range, y_range]}
            ROIs.append(region)

        cv2.destroyWindow(region_name)

    cap.release()
    cv2.destroyAllWindows()

    # Save ROIs to file
    with open(REGION_FILE, "w") as file:
        json.dump(ROIs, file)
    print("Regions saved successfully!")

    return ROIs


def load_saved_regions():
    """ Load saved regions from file if they exist. """
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE, "r") as file:
            return json.load(file)
    return None


def get_user_choice():
    """ Ask user whether to use saved regions or define new ones. """
    while True:
        choice = input("Use saved regions (Y) or define new ones (N)? ").strip().lower()
        if choice in ["y", "n"]:
            return choice
        print("Invalid input. Please enter 'Y' or 'N'.")

def get_region_from_position(center_x, center_y, regions):
    for region_id, region in enumerate(regions, start=1):
        x_range, y_range = region["range"]
        if x_range[0] <= center_x <= x_range[1] and y_range[0] <= center_y <= y_range[1]:
            return region_id
    return None

def draw_region_boundaries(frame, regions):
    """ Draws region boundaries on the frame. """
    for region in regions:
        points = np.array(region["polygon"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, region["name"], (points[0][0][0], points[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def detect_and_log_with_webcam(regions):
    """ Runs YOLO detection and logs movement data with anomaly detection. """
    conf_level = 0.25
    class_IDS = [0]  # Detect humans only
    log_file = "movement_log.csv"

    # Ensure the CSV file has a header
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["date", "time", "region"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] - Unable to access the webcam.")
        return

    print("[INFO] - Webcam detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] - Frame capture failed. Exiting.")
            break

        results = model_yolo(frame, conf=conf_level, classes=class_IDS, verbose=False)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        draw_region_boundaries(frame, regions)  # Draw regions on the frame

        for result in results:
            for bbox in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, bbox[:4])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                region_id = get_region_from_position(center_x, center_y, regions)
                if region_id is not None:
                    # Log to CSV file
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([date_str, time_str, REGION_MAPPING.get(region_id, 'Unknown')])

                    X_test = np.array([[now.hour, region_id]])  # Adjusted for input_size=2
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
                    print("region: {0} - hour: {1}".format(region_id, now.hour))
                    with torch.no_grad():
                        prediction = model_lstm(X_test_tensor)
                        prob = torch.sigmoid(prediction).item()

                    threshold = 0.4
                    is_anomaly = prob > threshold

                    color = (0, 255, 0) if not is_anomaly else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    if is_anomaly:
                        cv2.putText(frame,
                                    f"Anomalies Detected | {REGION_MAPPING.get(region_id, 'Unknown')} | {prob:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        cv2.putText(frame, f"{now.hour}|{REGION_MAPPING.get(region_id, 'Unknown')} | {prob:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # print(model_lstm.state_dict())

    saved_regions = load_saved_regions()
    if saved_regions:
        user_choice = get_user_choice()
        if user_choice == "y":
            regions = saved_regions
            print("Using saved regions.")
        else:
            regions = extract_roi_from_webcam(["Living Room", "Kitchen", "Bedroom"])
    else:
        print("No saved regions found. Extracting new regions.")
        regions = extract_roi_from_webcam(["Living Room", "Kitchen", "Bedroom"])

    detect_and_log_with_webcam(regions)
