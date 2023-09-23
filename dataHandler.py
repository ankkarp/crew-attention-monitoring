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
            unhandled = []
            for j in range(len(phones)):
                center = get_center(phones[j])
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
                    # frame, wrist pos, phone pos, index of human, phone_visibility, type of violation
                    violations.append([frame_id, time, [wrists], history[closest_id], i, visibility[closest_id]])

        # hidden phone between shoulders
        for i in range(len(history)):
            if visibility[i] == 0: # for each invisible phone
                center = get_center(history[i])

                for j in range(len(persons_shoulders)): # for each person
                    shoulders = persons_shoulders[j]
                    wrists = persons_wrists[j]
                    if shoulders[0][0] < center[0] < shoulders[1][0] and (
                        shoulders[0][0] < wrists[0][0] < shoulders[1][0] or shoulders[0][0] < wrists[1][0] < shoulders[1][0]
                    ):  # center of phone and one of the wrist between shoulders
                        violations.append([frame_id, time, [shoulders, wrists], history[i], j, 0])