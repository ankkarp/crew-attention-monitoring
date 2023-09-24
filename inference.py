# TEST YOUR MODEL HERE
from model import *
from analysis import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detection-model', help="Путь к весам модели детекции телефонов",
                    type=str, default='models/70_7_x_best.pt')
parser.add_argument('-p', '--pose-model', help="Путь к весам модели детекции поз",
                    type=str, default='models/yolov8x-pose.pt')
parser.add_argument('-v', '--video-path', help="Путь к исходному видео",
                    type=str, default='tests/test_video.mp4')
parser.add_argument('-o', '--out-path', help="Путь для сохранения обработанного видео",
                    type=str, default='results/test_result.avi')
parser.add_argument('-c', '--csv-save-path', help="Путь для сохранения результата работы модели",
                    type=str, default='csv_results/res_data.csv')

args = parser.parse_args()
csv_save_path = 'path to save result'

model = AttentionModel()
model.load_models(args.detection_model, args.pose_model)
model.process_video(args.video_path, args.out_path)

analyzer = Analyzer()
violations = analyzer.process_model_data(model.pos_est_data, model.detected_data)
ans = analyzer.process_violations(violations)


ans_df = pd.DataFrame(ans, columns=['Start', 'End'])
ans_df.to_csv(args.csv_save_path, index=False)

print(ans_df)
