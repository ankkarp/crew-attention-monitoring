import pandas as pd
import gradio as gr
from model import *
from analysis import *


df = pd.DataFrame(data={'Время': [1, 2], 'Нарушение': [3, 4]})

img = [r'C:\Users\renata\AppData\Local\Temp\gradio\my_screen\6375137857_721b45a66f_z.jpg', r'C:\Users\renata\Pictures\anon-анкета-анона-2163935.jpg']

# def some_model(video):
#     pred = model.predict(video)
#     imgs = your_func(pred, video)
#     return pred, imgs

def logic(video, text):
    # df, img = some_model(video)

    print('function call')

    if video:
        model = AttentionModel()
        model.load_models('', '')

        analyzer = Analyzer()
        violations = analyzer.process_model_data(model.pos_est_model, model.detected_data)
        ans = analyzer.process_violations(violations)




    #
    # if not text:
    #     text = 0
    # print(text)
    return video, df, [img[int(text)]], 'every'

inputs = [
    gr.Video(format='mp4'),
    gr.Textbox(lines=2, placeholder="0", label='Введите номер нарушения')
]


outputs = [
    "playable_video",
    gr.Dataframe(
        label="Результат обработки видео",
        # value=df,
        row_count=10,
        col_count=2,
    ),
    gr.Gallery(label="Фиксация нарушения", columns=(1, 3)),
    gr.CheckboxGroup([str(i) for i in range(df.shape[0])], label="Проверка", info="Было ли нарушение?")

]


demo = gr.Interface(
    logic,
    inputs=inputs,
    outputs=outputs,
)


demo.launch(share=True)