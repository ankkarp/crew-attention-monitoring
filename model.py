import time

import cv2
import pandas as pd
import torch
from tqdm import tqdm
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
    def __init__(self, batch_size=4):
        self.detected_history = []
        self.pos_est_values = []
        self.detected_values = []
        self.sec_per_frame = None
        self.batch_size = batch_size

        # models
        self.detect_model = None,
        self.detect_model_classes = None
        self.pos_est_model = None

        # video parameters
        self.video_type = 'avi'  # saved video type



    def load_models(self, detection_modelname=None, pose_estimator_name=None):
        if detection_modelname is not None:
            self.detect_model = YOLO(detection_modelname)
            self.detect_model_classes = self.detect_model.names
            self.detect_model.to('cuda')
        if pose_estimator_name:
            self.pos_est_model = YOLO(pose_estimator_name)
            self.pos_est_model.to('cuda')


    def handle_detection(self, results, frames, timestamps, frame_ids, save=False):
        detected_classes = []
        detected_positions = []

        for i in range(len(results)):
            boxes = results[i].boxes.cpu().numpy()

            for box in boxes:
                class_name = self.detect_model_classes[int(box.cls)]
                detected_classes.append(str(class_name))
                xyxy = box.xyxy[0]

                if save:
                    confidence = str(round(box.conf[0].item(), 2))
                    label = f'{class_name}: {confidence}'
                    frames[i] = plot_boxes(frames[i], xyxy, label)
                detected_positions.append(xyxy.tolist())

            if len(detected_classes) != 0:
                detection_time = frame_ids[i] * self.sec_per_frame
                self.detected_values.append([frame_ids[i], timestamps[i], detected_positions])
        return frames


    def handle_pos_est(self, results, frames, timestamps, frame_ids, save=False):
        for i in range(len(results)):
            keypoints = results[i].keypoints.xy.cpu().numpy().tolist()
            for pose in keypoints:
                if save:
                    for point in pose:
                        frames[i] = cv2.circle(frames[i], (int(point[0]), int(point[1])), radius=0, color=(0, 255, 0), thickness=10)
            self.pos_est_values.append([frame_ids[i], timestamps[i], len(results[i]), keypoints])
        return frames

    def process_batch(self, frames, timestamps, frame_ids, save):
        if self.pos_est_model is not None:
            pos_est_results = self.pos_est_model(frames, verbose=False)
            frames = self.handle_pos_est(pos_est_results, frames, timestamps, frame_ids, save)
        if self.detect_model is not None:
            detection_results = self.detect_model(frames, verbose=False)
            frames = self.handle_detection(detection_results, frames, timestamps, frame_ids, save)
        return frames

    def process_video(self, data_path, out_path, detection=True, pos_estimation=False, save=False):
        cap = cv2.VideoCapture(data_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.sec_per_frame = 1 / fps
        self.detected_values = []
        self.pos_est_values = []
        self.detected_history = []
        self.use_detection = detection
        self.use_pose_estimation = pos_estimation

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        codec = cv2.VideoWriter_fourcc('M','J','P','G')  # avi format
        out = cv2.VideoWriter(out_path, codec, fps * 2, (frame_width, frame_height))

        start = time.time()
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                frames = []
                frame_ids = []
                timestamps = []
                for i in range(self.batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                    frame_ids.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    frames.append(frame)
                if not success:
                    break
                if len(frames) != 0:
                    frames = self.process_batch(frames, timestamps, frame_ids, save)
                    if save:
                        for frame in frames:
                            out.write(frame)
                pbar.update(len(frames))
        end = time.time() - start
        print(f'Time: {end}')
        self.pos_est_data = pd.DataFrame(self.pos_est_values, columns=['Frame_id', 'Time', 'People_count', 'KeyPoints'])
        self.detected_data = pd.DataFrame(self.detected_values, columns=['Frame_id', 'Time', 'Position'])

    def get_detected_data(self):
        return self.detected_data

    def clear_data(self):  # clear detected_data
        self.detected_data = self.detected_data.iloc[0:0]
