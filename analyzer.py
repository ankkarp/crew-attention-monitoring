import numpy as np
from tqdm import tqdm


class Analyzer:
    def __init__(self, max_phone_movement, max_wrist_to_phone_distance=1080 / 2):
        self.history = []
        self.max_allowed_dist = max_phone_movement
        self.wrist_phone_dist = max_wrist_to_phone_distance
        self.violations = []

    def _get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def _trace_phones(self, phones):
        self.visibility = [0 for _ in range(len(self.history))]
        phones = self._update_phones(phones)
        if phones:
            self.history.extend(phones)
            self.visibility.extend([1 for i in range(len(phones))])  # we can see all extended phones

    def _update_phones(self, phones):
        if len(self.history) < len(phones):
            for i in range(len(self.history)):
                center = self._get_center(self.history[i])
                dists = list(map(lambda x: np.linalg.norm(self._get_center(x) - center), phones))
                min_dist = min(dists)

                if min_dist < self.max_allowed_dist:
                    closest_id = dists.index(min_dist)
                    self.history[i] = phones[closest_id]
                    self.visibility[i] = 1  # we can see phone on frame
                    phones.pop(closest_id)

            if phones:
                self.history.extend(phones)
                self.visibility.extend([1 for i in range(len(phones))])  # we can see all extended phones

        else:
            unhandled = []
            for j in range(len(phones)):
                center = self._get_center(phones[j])
                dists = list(map(lambda x: np.linalg.norm(self._get_center(x) - center), self.history))
                min_dist = min(dists)

                if min_dist < self.max_allowed_dist:
                    closest_id = dists.index(min_dist)
                    self.history[closest_id] = phones[j]
                    self.visibility[closest_id] = 1
                else:
                    unhandled.append(phones[j])
            if unhandled:
                self.history.extend(unhandled)
                self.visibility.extend([1 for i in range(len(unhandled))])  # we can see all extended phones
        return phones

    def _generate_violations(self):
        violations_processed = []
        passed_time = 0
        start_time = -1

        for i in range(0, len(self.violations) - 1):
            time_diff = self.violations[i + 1][1] - self.violations[i][1]
            print(time_diff)
            if self.violations[i + 1][4] == self.violations[i][4] and time_diff <= 1:
                if start_time == -1:
                    start_time = self.violations[i][1]
                passed_time += time_diff

            else:
                print(passed_time, 'passed_time')
                if passed_time > 1:
                    # может прерываться видимость, сделать проверку
                    violations_processed.append([self.violations[i][4], start_time, self.violations[i][1], self.violations[i][5]])

                passed_time = 0
                start_time = -1
                
    def _wrist_to_phone_distance(self, persons_wrists, frame_id, time):
        for i in range(len(persons_wrists)):
            for wrists in persons_wrists[i]:
                dists = list(map(lambda x: np.linalg.norm(self._get_center(x) - wrists), self.history))
                min_dist = min(dists)
                closest_id = dists.index(min_dist)
                if min_dist <= self.wrist_phone_dist:
                    # frame, wrist pos, phone pos, index of human, phone_visibility, type of violation
                    self.violations.append(
                        [frame_id, time, [wrists], self.history[closest_id], i, self.visibility[closest_id]])

    def _wrist_to_phone_distance(self, persons_wrists, frame_id, time):
        for i in range(len(persons_wrists)):
            for wrists in persons_wrists[i]:
                dists = list(map(lambda x: np.linalg.norm(self._get_center(x) - wrists), self.history))
                min_dist = min(dists)
                closest_id = dists.index(min_dist)
                if min_dist <= self.wrist_phone_dist:
                    # frame, wrist pos, phone pos, index of human, phone_visibility, type of violation
                    self.violations.append(
                        [frame_id, time, [wrists], self.history[closest_id], i, self.visibility[closest_id]])

    def _get_phone_behind_shoulders(self, persons_shoulders, persons_wrists, frame_id, time):
        for i in range(len(self.history)):
            if self.visibility[i] == 0:  # for each invisible phone
                center = self._get_center(self.history[i])

                for j in range(len(persons_shoulders)):  # for each person
                    shoulders = persons_shoulders[j]
                    wrists = persons_wrists[j]
                    if frame_id == 2188:
                        pass
                    if len(shoulders) != 0 and shoulders[0][0] < center[0] < shoulders[1][0] and (
                            shoulders[0][0] < wrists[0][0] < shoulders[1][0] or shoulders[0][0] < wrists[1][0] <
                            shoulders[1][0]
                    ):  # center of phone and one of the wrist between shoulders
                        self.violations.append([frame_id, time, [shoulders, wrists], self.history[i], j, 0])


    def get_violations(self, pos_est_data, detection_data):
        for frame_id in tqdm(pos_est_data.index):
            persons_wrists = [pose[9:11] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
            persons_shoulders = [pose[5:7] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
            time = pos_est_data.loc[frame_id, 'Time']
            if frame_id in detection_data.index:  # we need to update phones last positions
                phones = detection_data.loc[frame_id, 'Position'].copy()
                self._trace_phones(phones)
            if self.history:
                # wrist near phone
                self._wrist_to_phone_distance(persons_wrists, frame_id, time)
                # hidden phone between shoulders
                self._get_phone_behind_shoulders(persons_shoulders, persons_wrists, frame_id, time)

