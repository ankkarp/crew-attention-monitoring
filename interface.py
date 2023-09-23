import pandas as pd
import gradio as gr
from model import *
from analysis import *
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
    gr.Gallery(label="Фиксация нарушения"),
    gr.Dataframe(
        label="Результат обработки видео",
        row_count=10,
        max_rows=10,
        col_count=2,
    ),
]


def logic(video_path, image_time):
    pd_ans = None
    image = None

    print(image_time)

    if video_path:
        model = AttentionModel()
        model.load_models(args.detection_model, args.pose_model)
        model.process_video(video_path, 'temp/output_video.avi')

        analyzer = Analyzer()
        violations = analyzer.process_model_data(model.pos_est_model, model.detected_data)
        ans = analyzer.process_violations(violations)

        pd_ans = pd.DataFrame(ans, columns=['Start_time', 'End_time'])

    if image_time and video_path:
        image = save_handled_timestamp(image_time, video_path, 'temp')

    return image, pd_ans


demo = gr.Interface(
    logic,
    inputs=inputs,
    outputs=outputs,
)

demo.launch(share=True)
