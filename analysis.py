import numpy as np
from tqdm import tqdm


class Analyzer:
    def __init__(self, min_wrist_dist=110, max_wrist_dist=130, max_wrist_move=10):
        self.min_wrist_dist = min_wrist_dist
        self.max_wrist_dist = max_wrist_dist
        self.max_wrist_move = max_wrist_move
        self.violations = []

    def _get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def process_model_data(self, pos_est_data, detection_data):
        violations = []

        for frame_id in pos_est_data.index:
            persons_wrists = [pose[9:11] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
            time = pos_est_data.loc[frame_id, 'Time']

            if frame_id in detection_data.index:
                phones = detection_data.loc[frame_id, 'Position'].copy()

                for i in range(len(persons_wrists)):

                    for j in range(len(persons_wrists[i])):
                        dists = list(map(lambda x: np.linalg.norm(self._get_center(x) - persons_wrists[i][j]), phones))
                        min_dist = min(dists)

                        closest_id = dists.index(min_dist)
                        closest_phone = phones[closest_id]

                        if min_dist <= self.max_wrist_dist:
                            if not ((j == 0 and persons_wrists[i][j][0] > closest_phone[2])
                                    or (j == 1 and persons_wrists[i][j][0] < closest_phone[0])):
                                violations.append([frame_id, time])
        return violations

    def _update_violations(self, violations_processed, passed_time, violation, start_time):
        if passed_time > 3000 and violation[2]:  # min_wrist_dist
            total_sec = start_time / 1000
            min = int(total_sec // 60)
            sec = int(total_sec % 60)

            total_p_sec = passed_time / 1000
            p_min = int(total_p_sec // 60)
            p_sec = int(total_p_sec % 60)

            violations_processed.append(f'{min}:{sec}, passed_time: {p_min}:{p_sec}')
        return violations_processed

    def process_violations(self, violations):
        violations_processed = []
        passed_time = 0
        start_time = -1

        for i in range(0, len(violations) - 1):
            time_diff = violations[i + 1][1] - violations[i][1]

            if (passed_time < 3000 and time_diff <= 2000) or (passed_time > 3000 and time_diff <= 10000):
                if start_time == -1:
                    start_time = violations[i][1]
                passed_time += time_diff

                if i == len(violations) - 2:

                    if passed_time > 3000:
                        total_sec = start_time / 1000
                        min = int(total_sec // 60)
                        sec = int(total_sec % 60)

                        total_p_sec = (start_time + passed_time) / 1000
                        p_min = int(total_p_sec // 60)
                        p_sec = int(total_p_sec % 60)

                        violations_processed.append(f'{min}:{sec}, end: {p_min}:{p_sec}')
            else:
                if passed_time > 3000:
                    total_sec = start_time / 1000
                    min = int(total_sec // 60)
                    sec = int(total_sec % 60)

                    total_p_sec = (start_time + passed_time) / 1000
                    p_min = int(total_p_sec // 60)
                    p_sec = int(total_p_sec % 60)

                    violations_processed.append(f'start: {min}:{sec}, end: {p_min}:{p_sec}')

                passed_time = 0
                start_time = -1
        return violations_processed

    def get_violations(self, pos_est_data, detection_data):
        violations = self.process_model_data(pos_est_data, detection_data)
        return self.process_violations(violations)