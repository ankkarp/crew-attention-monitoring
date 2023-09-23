import pandas as pd
import gradio as gr
from model import *
from analysis import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--detection-model', help="Путь к весам модели детекции телефонов",
                    type=str, default='models/70_7_x_best.pt')
parser.add_argument('-p', '--pose-model', help="Путь к весам модели детекции поз",
                    type=str, default='models/yolov8x-pose.pt')
args = parser.parse_args()


inputs = [
    gr.Video(format='mp4'),
    # gr.Textbox(lines=2, placeholder="0", label='Введите номер нарушения')
]

outputs = [
    "playable_video",
    gr.Dataframe(
        label="Результат обработки видео",
        # value=df,
        max_rows=10,
        col=2,
    ),
    # gr.Gallery(label="Фиксация нарушения", columns=(1, 3)),
    # gr.CheckboxGroup([str(i) for i in range(df.shape[0])], label="Проверка", info="Было ли нарушение?")
]


def logic(video):
    type(video)
    # df, img = some_model(video)
    pd_ans = None

    if video:

        model = AttentionModel()
        model.load_models(args.detection_model, args.pose_model)
        model.process_video('data_path', 'out_path')

        analyzer = Analyzer()
        violations = analyzer.process_model_data(model.pos_est_model, model.detected_data)
        ans = analyzer.process_violations(violations)

        pd_ans = pd.DataFrame(ans, columns=['Start_time', 'End_time'])

    return video, pd_ans


demo = gr.Interface(
    logic,
    inputs=inputs,
    outputs=outputs,
)

demo.launch(share=True)
