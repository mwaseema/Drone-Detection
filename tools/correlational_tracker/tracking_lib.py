import os

def format_frame_number(frame_number: int):
    return '{:0>6}'.format(frame_number)

def get_bounding_boxes(detections_path: str, video_name: str, frame_number: int):
    frame_number_formatted = format_frame_number(frame_number)
    detections_file_path = os.path.join(detections_path, f'{video_name}_{frame_number_formatted}.txt')

    if not os.path.exists(detections_file_path):
        return None
    else:
        fl = open(detections_file_path)
        lines = fl.readlines()

        bboxes = []

        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            line_split = lines[i].split(' ')

            x1 = int(line_split[2])
            y1 = int(line_split[3])
            x2 = int(line_split[4])
            y2 = int(line_split[5])
            w = x2 - x1
            h = y2 - y1

            bbox = (x1, y1, w, h)
            bboxes.append(bbox)

        return bboxes
