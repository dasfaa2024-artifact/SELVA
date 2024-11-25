import pickle
from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd


def _greater_than_2_conf(df):
    r"""
    suppose k = len(df), using formular:
    $1 - \prod_{i}^{k}{1-p(x_i)} - \sum_{i}^{k}{p(x_i)\cdot\prod_{j\neq i}^{k}{1-p(x_i)}}$
    :param df:
    :return:
    """
    conf = df['conf'].to_numpy()
    neg_conf = 1 - conf
    none = np.prod(neg_conf)
    only_one = np.sum(conf / neg_conf) * none
    return 1 - none - only_one


def both_exit(df, cls_lst):
    labels = df.labels
    mask = np.isin(cls_lst, labels)
    if mask.all():
        confs = df.conf
        return confs[labels == cls_lst[0]].max() * confs[labels == cls_lst[1]].max()
    else:
        return 0.


def _greater_than_1_conf(df):
    conf = df['conf'].to_numpy()
    neg_conf = 1 - conf
    none = np.prod(neg_conf)
    return 1 - none


def predict_reduce(model_result, name_ls, cls_desideratum, obj_num, filter_func, save_path=None):
    res = {}

    for n in name_ls:
        objs = model_result[n]
        end = round(objs[-1, 0]) + 1
        if isinstance(cls_desideratum, Iterable):
            header = ['fid', 'labels', 'conf']
            objs = objs[np.isin(objs[:, 1], cls_desideratum)]
            dtype = (np.int32, np.float32, np.float32)
        else:
            header = ['fid', 'conf']
            dtype = (np.int32, np.float32)
            objs = objs[objs[:, 1] == cls_desideratum][:, [0, -1]]
        df = pd.DataFrame(data=objs, columns=header).astype(dict(zip(header, dtype)))
        # object detection filter: at least 2 people
        df = df.groupby('fid').filter(lambda x: x.fid.count() >= obj_num)
        if filter_func is not None:
            df = df.groupby('fid').apply(filter_func)
        else:
            df = df.groupby('fid').max()
        # fill other frames with conf 0
        df = df.conf if isinstance(df, pd.DataFrame) else df

        df = df.reindex(np.arange(end), fill_value=0)
        res[n] = df
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(res, f)
    return res


def threshold_cut(p, th=0.5):
    return np.where(p > th, 1, 0)



if __name__ == '__main__':
    # with open('/home/lg/VDBM/multi_predicate/selectivity_sampling'
    #           '/grouped_label.pkl', 'rb') as f:

    with open('/home/lg/VDBM/multi_predicate/selectivity_sampling'
              '/query/angry_bernie_emotion01.pkl', 'rb') as f:
        data = pickle.load(f)
    pseudo_label = {}
    predict_all = {}
    # face detect
    # facedetect = ['fd', 'dffd'] + [f'{abr}_tasti' for abr in ['dffd', 'fd']]
    # # facedetect = [f'{abr}_tasti' for abr in ['dffd', 'fd']]
    # filter_cls = np.array([-1, 3485])
    # # func = partial(both_exit, cls_lst=filter_cls)
    # # predict_all.update(predict_reduce(dict(data['face']), facedetect, filter_cls, 2, func))
    # predict_all.update(predict_reduce(dict(data['face']), facedetect,
    #                                   filter_cls, 2,
    #             lambda df: 1. if np.isin(filter_cls, df.labels).all() else 0))
    # emotion detect
    # emo = ['ferc', 'ferm', 'dfe'] + [f'{abr}_tasti' for abr in ['ferm', 'dfe', 'ferc']]
    emo = ['ferc', 'ferm']
    # emo = [f'{abr}_tasti' for abr in ['ferm', 'dfe']]
    filter_cls = 0
    # predict_all.update(predict_reduce(dict(data['emotion']), emo, filter_cls, 2, _greater_than_1_conf))
    predict_all.update(predict_reduce(dict(data), emo, filter_cls,
                                      1, None))
    # # object detect
    # yolo_series = ['n', 's', 'm', 'l', 'x'] + [f'{abr}_tasti' for abr in ['n', 's', 'm', 'l', 'x']]
    # filter_cls = 0
    # # predict_all.update(predict_reduce(dict(data['detect']), yolo_series, filter_cls, 2, _greater_than_2_conf))
    # predict_all.update(predict_reduce(dict(data['detect']), yolo_series,
    #                                   filter_cls, 2, None))
    with open('filtered_emotion.pkl', 'wb') as f:
        pickle.dump(predict_all, f)
    for name in predict_all:
        pseudo_label[name] = threshold_cut(predict_all[name], 0.5)
    with open('pseudo_label_emotion.pkl', 'wb') as f:
        pickle.dump(pseudo_label, f)
