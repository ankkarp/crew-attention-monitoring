import cv2


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

def save_handled_timestamp(timestamp, saved_video_path, img_save_path):
    cap = cv2.VideoCapture(saved_video_path)
    success, frame = cap.read()

    while cap.get(cv2.CAP_PROP_POS_MSEC) < timestamp:
        success, frame = cap.read()

    cv2.imwrite(f'{img_save_path}/detection_frame_{timestamp}.jpg', frame)
