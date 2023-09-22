import time

import cv2
import pandas as pd
import torch
from ultralytics import YOLO


def plot_boxes(frame, xyxy, label):  # plot detected class box
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    frame = cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0,0,255), -1)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    frame = cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

def save_handled_frame(frame_id, saved_video_path, img_save_path):
    cap = cv2.VideoCapture(saved_video_path)
    count = 1
    success, frame = cap.read()

    while count != frame_id:
        count += 1
        success, frame = cap.read()

    cv2.imwrite(f'{img_save_path}/detection_frame_{frame_id}.jpg', frame)
    print('image was saved')


class AttentionModel:
    def __init__(self, detection_model='yolov8x.pt'):
        self.models_keys = {
            'detection': detection_model,
            'pos_estimation': 'models/yolov8x-pose.pt'
        }

        # models
        self.detect_model = None,
        self.detect_model_classes = None
        self.pos_est_model = None

        self.detected_data = pd.DataFrame(
            columns=['Frame_id', 'Time', 'Objects_class', 'Position'],
        )
        self.detected_values = []
        self.pos_est_data = pd.DataFrame(
            columns=['Frame_id', 'Time', 'Keypoints'],
        )
        self.pos_est_values = []

        # video parameters
        self.video_type = 'avi'  # saved video type
        self.sec_per_frame = None


    def load_models(self):
        self.detect_model = YOLO(self.models_keys['detection'])
        self.detect_model_classes = self.detect_model.names

        self.pos_est_model = YOLO(self.models_keys['pos_estimation'])

        if torch.cuda.is_available():
            self.detect_model.to('cuda')
            self.pos_est_model.to('cuda')


    def handle_detection(self, results, frame, frame_id, save=False):
        detected_classes = []
        detected_positions = dict()

        for result in results:
            boxes = result.boxes.cpu().numpy()

            for box in boxes:
                class_name = self.detect_model_classes[int(box.cls)]
                detected_classes.append(str(class_name))
                xyxy = box.xyxy[0]

                if save:
                    confidence = str(round(box.conf[0].item(), 2))
                    label = f'{class_name}: {confidence}'
                    frame = plot_boxes(frame, xyxy, label)

                if not class_name in detected_positions.keys():
                    detected_positions[class_name] = []
                detected_positions[class_name].append(xyxy)

        detected_classes = set(detected_classes)

        if detected_classes:
            detection_time = frame_id * self.sec_per_frame
            self.detected_values.append([frame_id, detection_time, detected_classes, detected_positions])
        return frame


    def handle_pos_est(self, results, frame, frame_id, save=False):
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()[0]

            if save:
                for point in keypoints:
                    frame = cv2.circle(frame, (int(point[0]), int(point[1])), radius=0, color=(0, 255, 0), thickness=10)

            detection_time = frame_id * self.sec_per_frame
            self.pos_est_values.append([frame_id, detection_time, keypoints])
        return frame


    def process_video(self, data_path, out_path, detection=True, pos_estimation=False, save=False):
        cap = cv2.VideoCapture(data_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.sec_per_frame = 1 / fps


        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        codec = cv2.VideoWriter_fourcc('M','J','P','G')  # avi format
        out = cv2.VideoWriter(out_path, codec, fps * 2, (frame_width, frame_height))

        start = time.time()

        success, frame = cap.read()
        frame_count = 0

        while success:
            frame_count += 1

            if detection:
                detection_results = self.detect_model(frame, verbose=False)
                frame = self.handle_detection(detection_results, frame, frame_count, save)

            if pos_estimation:
                pos_est_results = self.pos_est_model(frame)
                frame = self.handle_pos_est(pos_est_results, frame, frame_count, save)

            if save:
                out.write(frame)
            success, frame = cap.read()
        end = time.time() - start
        print(f'Time: {end}')
        self.pos_est_data = pd.DataFrame(self.pos_est_values, columns=['Frame_id', 'Time', 'Keypoints'])
        self.detected_data = pd.DataFrame(self.detected_values, columns=['Frame_id', 'Time', 'Objects_class', 'Position'])

    def get_detected_data(self):
        return self.detected_data

    def clear_data(self):  # clear detected_data
        self.detected_data = self.detected_data.iloc[0:0]
