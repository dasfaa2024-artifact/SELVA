import os
import pickle

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_batch_frame(video_path, bs):
    if not os.path.exists(video_path):
        raise FileExistsError(f'{video_path} not exist')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ConnectionError(f'{video_path} failed to open')
    count = 0
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        img = [info[1] for _ in range(bs) if (info := cap.read())[0]]
        read_num = len(img)
        if read_num < bs:
            cap.release()
            if read_num:
                yield img
                print(f'read {count + read_num}/{frame_num} of {video_path}.')
            break
        else:
            yield img
            count += bs
            if count % 32 == 0:
                print(f'read {count}/{frame_num} of {video_path}.')


def yolo_result(result, ide):
    boxes = result.boxes
    if boxes.shape[0] != 0:
        # xyxy, conf, cls
        data = boxes.input[:, [-1, - 2] + list(range(4))].cpu().numpy().astype(np.float32)
        data = np.insert(data, 0, ide, axis=1)
    else:
        data = None
    return data


def detect_video(video_file, models, img_sz, bs, save_path=None):
    """
    detect objects in video file
    :param video_file:  path of video file
    :param models: detect models, a dict of {model name: model}.
    :param img_sz: input image size for detection
    :param bs: batch size
    :param save_path: where to save detection results
    :return: dictionary map model to detection results
    """
    model_sample = next(iter(models.values()))
    model_sample.predict(np.empty((*img_sz, 3), dtype=np.uint8), imgsz=img_sz)
    preprocessor = model_sample.predictor.preprocess

    batcher = get_batch_frame(video_file, bs)
    table = {name: [] for name in models}
    fid = 0
    with torch.no_grad():
        for imgs in batcher:
            imgs = preprocessor(imgs)
            for name, m in models.items():
                results = m.predict(imgs, imgsz=img_sz, verbose=False, stream=True)
                points = table[name]
                for i, res in enumerate(results):
                    objs = yolo_result(res, fid + i)
                    if objs is not None:
                        points.append(objs)
            fid += len(imgs)
            del imgs
    for k in table:
        table[k] = np.concatenate(table[k])

    if save_path is not None:
        pickle.dump(table, open(save_path, 'wb'))
    return table


if __name__ == '__main__':
    imgsz = (384, 640)
    batchsz = 16

    video_path = '/home/cw/VDMS/viva-vldb23-artifact/dataset/angrybernie/data/new_interview.mp4'

    test_img = ['']
    yolo_weight_root = '/home/cw/VDMS/multi-predicate/weights'
    out_put = '/home/cw/VDSM/multi-predicate/yolo_result.pkl'
    yolo_series = ['n', 's', 'm', 'l', 'x']
    yolo_models = {abr: YOLO(os.path.join(yolo_weight_root, f'yolov5{abr}u.pt')) for abr in yolo_series}
    detect_video(video_path, yolo_models, imgsz, batchsz, out_put)
