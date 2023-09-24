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

    def _update_violations(self, violations_processed, passed_time, start_time):
        if passed_time > 3000:
            total_sec = start_time / 1000
            min = int(total_sec // 60)
            sec = int(total_sec % 60)

            total_p_sec = (start_time + passed_time) / 1000
            p_min = int(total_p_sec // 60)
            p_sec = int(total_p_sec % 60)

            violations_processed.append([f'{min}:{sec:02}', f'{p_min}:{p_sec:02}'])
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
                    violations_processed = self._update_violations(violations_processed, passed_time, start_time)
            else:
                violations_processed = self._update_violations(violations_processed, passed_time, start_time)

                passed_time = 0
                start_time = -1
        return violations_processed

    def full_process_model_data(self, pos_est_data, detection_data):
        history = []
        max_allowed_dist = 1080 / 2
        wrist_phone_dist = 90
        violations = []

        for frame_id in pos_est_data.index:
            print(frame_id, len(violations), len(history))

            persons_wrists = [pose[9:11] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
            persons_shoulders = [pose[5:7] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
            time = pos_est_data.loc[frame_id, 'Time']

            if frame_id in detection_data.index:  # we need to update phones last positions
                phones = detection_data.loc[frame_id, 'Position'].copy()
                visibility = [0 for i in range(len(history))]

                if len(history) < len(phones):
                    for i in range(len(history)):
                        center = self._get_center(history[i])
                        dists = list(map(lambda x: np.linalg.norm(get_center(x) - center), phones))
                        min_dist = min(dists)

                        if min_dist < max_allowed_dist:
                            closest_id = dists.index(min_dist)
                            history[i] = phones[closest_id]
                            visibility[i] = 1  # we can see phone on frame
                            phones.pop(closest_id)

                    if phones:
                        history.extend(phones)
                        visibility.extend([1 for i in range(len(phones))])  # we can see all extended phones

                else:
                    unhandled = []
                    for j in range(len(phones)):
                        center = self._get_center(phones[j])
                        dists = list(map(lambda x: np.linalg.norm(get_center(x) - center), history))
                        min_dist = min(dists)

                        if min_dist < max_allowed_dist:
                            closest_id = dists.index(min_dist)
                            history[closest_id] = phones[j]
                            visibility[closest_id] = 1
                        else:
                            unhandled.append(phones[j])
                    if unhandled:
                        history.extend(unhandled)
                        visibility.extend([1 for i in range(len(unhandled))])  # we can see all extended phones

            # if we have info about phones pos
            if history:
                # wrist near phone
                for i in range(len(persons_wrists)):
                    for wrists in persons_wrists[i]:
                        dists = list(map(lambda x: np.linalg.norm(get_center(x) - wrists), history))
                        min_dist = min(dists)
                        closest_id = dists.index(min_dist)
                        if min_dist <= wrist_phone_dist:
                            violations.append([frame_id, time, visibility[closest_id]])

                # hidden phone between shoulders
                for i in range(len(history)):
                    if visibility[i] == 0:  # for each invisible phone
                        center = self._get_center(history[i])

                        for j in range(len(persons_shoulders)):  # for each person
                            shoulders = persons_shoulders[j]
                            wrists = persons_wrists[j]
                            if shoulders[0][0] < center[0] < shoulders[1][0] and (
                                    shoulders[0][0] < wrists[0][0] < shoulders[1][0] or shoulders[0][0] < wrists[1][0] <
                                    shoulders[1][0]
                            ):
                                violations.append([frame_id, time, 0])
        return violations

    def get_violations(self, pos_est_data, detection_data):
        violations = self.process_model_data(pos_est_data, detection_data)
        return self.process_violations(violations)