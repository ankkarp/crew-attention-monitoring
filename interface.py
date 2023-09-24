import gradio as gr

from analysis import Analyzer
from model import *
import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--detection-model', help="Путь к весам модели детекции телефонов",
                    type=str, default='models/70_7_x_best.pt')
parser.add_argument('-p', '--pose-model', help="Путь к весам модели детекции поз",
                    type=str, default='models/yolov8x-pose.pt')
args = parser.parse_args()


inputs = [
    gr.Video(label='Input Video'),
    gr.Textbox(lines=1, placeholder="0:00", label='Введите время нарушения')
]

outputs = [
    gr.Image(type='numpy'),
    gr.Dataframe(
        label="Результат обработки видео",
        row_count=10,
        max_rows=10,
        col_count=2,
    ),
]

pd_ans = pd.DataFrame(columns=['Start_time', 'End_time'])
last_video = None

def logic(video_path, image_time):
    global last_video
    global pd_ans
    image = None
    print(image_time)
    if video_path and last_video != video_path:
        pd_ans = pd.DataFrame(columns=['Start_time', 'End_time'])
        model = AttentionModel()
        model.load_models(args.detection_model, args.pose_model)
        model.process_video(video_path, 'result/output_video.avi')

        analyzer = Analyzer()
        violations = analyzer.process_model_data(model.pos_est_data, model.detected_data)
        ans = analyzer.process_violations(violations)

        pd_ans = pd.DataFrame(ans, columns=['Start_time', 'End_time'])
        last_video = video_path

    if image_time and video_path:
        time_ms = sum([int(el) * 60 ** i for i, el in enumerate(image_time.split(':')[::-1])]) * 1000
        image = save_handled_timestamp(time_ms, 'result/output_video.avi')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, pd_ans


demo = gr.Interface(
    logic,
    inputs=inputs,
    outputs=outputs,
)

demo.launch(share=True)
