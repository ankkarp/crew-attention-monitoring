pos_est_data = model.pos_est_data.drop_duplicates(subset=['Frame_id'])
detection_data = model.detected_data.set_index('Frame_id')
pos_est_data = pos_est_data.set_index('Frame_id')


history = []
max_allowed_dist = 1080 / 2
wrist_phone_dist = 90
violations = []


def get_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


for frame_id in pos_est_data.index:
    persons_wrists = [pose[9:11] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
    persons_shoulders = [pose[5:7] for pose in pos_est_data.loc[frame_id, 'KeyPoints']]
    time = pos_est_data.loc[frame_id, 'Time']

    if frame_id in detection_data.index:  # we need to update phones last positions
        phones = detection_data.loc[frame_id, 'Position'].copy()
        visibility = [0 for i in range(len(history))]

        if len(history) < len(phones):
            for i in range(len(history)):
                center = get_center(history[i])
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
            for j in range(len(phones)):
                center = get_center(history[j])
                dists = list(map(lambda x: np.linalg.norm(get_center(x) - center), history))
                min_dist = min(dists)

                if not min_dist > max_allowed_dist:
                    closest_id = dists.index(min_dist)
                    history[j] = phones[closest_id]
                    visibility[j] = 1  # we can see phone on frame
                    phones.pop(closest_id)
            if phones:
                history.extend(phones)
                visibility.extend([1 for i in range(len(phones))])  # we can see all extended phones

        #print(history, frame_id)

    # if we have info about phones pos
    if history:
        # wrist near phone
        for i in range(len(persons_wrists)):
            for wrists in persons_wrists[i]:
                dists = list(map(lambda x: np.linalg.norm(get_center(x) - wrists), history))
                min_dist = min(dists)
                closest_id = dists.index(min_dist)
                if min_dist <= wrist_phone_dist:
                    # frame, wrist pos, phone pos, index of human, phone_visibility, type of violation
                    violations.append([frame_id, time, [wrists], history[closest_id], i, visibility[closest_id]])

        # hidden phone between shoulders
        for i in range(len(history)):
            if visibility[i] == 0:  # for each invisible phone
                center = get_center(history[i])

                for j in range(len(persons_shoulders)): # for each person
                    shoulders = persons_shoulders[j]
                    wrists = persons_wrists[j]
                    if shoulders[0][0] < center[0] < shoulders[1][0] and (
                        shoulders[0][0] < wrists[0][0] < shoulders[1][0] or shoulders[0][0] < wrists[1][0] < shoulders[1][0]
                    ):  # center of phone and one of the wrist between shoulders
                        violations.append([frame_id, time, [shoulders, wrists], history[i], j, 0])


violations_processed = []
passed_time = 0
start_time = -1

count = 0
for i in range(0, len(violations) - 1):
    time_diff = violations[i + 1][1] - violations[i][1]
    print(time_diff)
    if violations[i + 1][4] == violations[i][4] and time_diff <= 1:
        if start_time == -1:
            start_time = violations[i][1]
        passed_time += time_diff

    else:
        print(passed_time, 'passed_time')
        if passed_time > 1:
            # может прерываться видимость, сделать проверку
            violations_processed.append([violations[i][4], start_time, violations[i][1], violations[i][5]])

        passed_time = 0
        start_time = -1