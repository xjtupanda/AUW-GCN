import os
import glob 
import random
from random import shuffle

import pandas as pd
import yaml
import numpy as np
from scipy.stats import norm
import torch
from torch.utils import data

from tool import set_seed


class LOSO_DATASET(data.Dataset):
    def __init__(self, opt, split, subject):
        super().__init__()
        self._split = split
        self._subject = subject
        self._segment_feat_root = opt['segment_feat_root']
        self._segment_len = opt['SEGMENT_LENTH']
        self._anno_df = pd.read_csv(opt['anno_csv'])
        self._step = int(opt["RECEPTIVE_FILED"] // 2)
        self._micro_normal_range = opt["micro_normal_range"]
        self._macro_normal_range = opt["macro_normal_range"]
        self._record_feat_path()

    def __len__(self):
        return len(self._feat_file_path_list)
    
    def __getitem__(self, index):
        feat_file_path = self._feat_file_path_list[index]
        data = np.load(feat_file_path)
        feature = data['feature']  # (256, 12, 2)
        labels = data['label']
        # get meta info
        vid_name = data['video_name'].item()
        offset = int(os.path.splitext(os.path.basename(feat_file_path))[0].split('_')[-1])
        
        STEP = self._step
        _macro_normal_range = self._macro_normal_range
        _micro_normal_range = self._micro_normal_range

        micro_start_labels = labels[:, 0]
        micro_apex_labels = labels[:, 1]
        micro_end_labels = labels[:, 2]
        micro_action_labels = labels[:, 3]

        macro_start_labels = labels[:, 4]
        macro_apex_labels = labels[:, 5]
        macro_end_labels = labels[:, 6]
        macro_action_labels = labels[:, 7]

        micro_start_end_label = [2] * (self._segment_len + STEP * 2)  # 0->start, 1->end, 2->None
        macro_start_end_label = [2] * (self._segment_len + STEP * 2)  # 0->start, 1->end, 2->None
        micro_apex_score = [0.] * (self._segment_len + STEP * 2)
        macro_apex_score = [0.] * (self._segment_len + STEP * 2)
        micro_action_score = [0.] * (self._segment_len + STEP * 2)
        macro_action_score = [0.] * (self._segment_len + STEP * 2)

        # start and end (micro)
        for index, start_label in enumerate(micro_start_labels):
            if start_label == 1:
                micro_start_end_label[index] = 0
                for j in range(1, _micro_normal_range + 1):
                    if index - j >= 0:
                        micro_start_end_label[index - j] = 0
                    if index + j < self._segment_len:
                        micro_start_end_label[index + j] = 0
        for index, end_label in enumerate(micro_end_labels):
            if end_label == 1:
                micro_start_end_label[index] = 1
                for j in range(1, _micro_normal_range + 1):
                    if index - j >= 0:
                        micro_start_end_label[index - j] = 1
                    if index + j < self._segment_len:
                        micro_start_end_label[index + j] = 1

        # start and end (macro)
        for index, start_label in enumerate(macro_start_labels):
            if start_label == 1:
                macro_start_end_label[index] = 0
                for j in range(1, _macro_normal_range + 1):
                    if index - j >= 0:
                        macro_start_end_label[index - j] = 0
                    if index + j < self._segment_len:
                        macro_start_end_label[index + j] = 0
        for index, end_label in enumerate(macro_end_labels):
            if end_label == 1:
                macro_start_end_label[index] = 1
                for j in range(1, _macro_normal_range + 1):
                    if index - j >= 0:
                        macro_start_end_label[index - j] = 1
                    if index + j < self._segment_len:
                        macro_start_end_label[index + j] = 1
        
        
        
        
        
        # apex (micro)
        rv = norm(loc=0, scale=4)  # 8
        for index, apex_label in enumerate(micro_apex_labels):
            if apex_label == 1:
                micro_apex_score[index] = 1
                for j in range(1, _micro_normal_range + 1):
                    if index - j > 0:
                        micro_apex_score[index - j] = rv.pdf(j) / rv.pdf(0)
                    if index + j < self._segment_len:
                        micro_apex_score[index + j] = rv.pdf(j) / rv.pdf(0)
        # apex (macro)
        for index, apex_label in enumerate(macro_apex_labels):
            if apex_label == 1:
                macro_apex_score[index] = 1
                for j in range(1, _macro_normal_range + 1):
                    if index - j > 0:
                        macro_apex_score[index - j] = rv.pdf(j) / rv.pdf(0)
                    if index + j < self._segment_len:
                        macro_apex_score[index + j] = rv.pdf(j) / rv.pdf(0)

        # action score
        for index, action_label in enumerate(micro_action_labels):
            if action_label == 1:
                micro_action_score[index] = 1
        for index, action_label in enumerate(macro_action_labels):
            if action_label == 1:
                macro_action_score[index] = 1

        micro_start_end_label = micro_start_end_label[STEP:-STEP]
        macro_start_end_label = macro_start_end_label[STEP:-STEP]
        micro_apex_score = micro_apex_score[STEP:-STEP]
        macro_apex_score = macro_apex_score[STEP:-STEP]
        micro_action_score = micro_action_score[STEP:-STEP]
        macro_action_score = macro_action_score[STEP:-STEP]

        feature = torch.tensor(feature).float()
        feature[:, :, 0] = (feature[:, :, 0] - 0.003463) / 0.548588
        feature[:, :, 1] = (feature[:, :, 1] - 0.003873) / 0.645621
        micro_start_end_label = torch.tensor(micro_start_end_label,
                                             dtype=torch.int64)
        macro_start_end_label = torch.tensor(macro_start_end_label,
                                             dtype=torch.int64)
        micro_apex_score = torch.tensor(micro_apex_score).float()
        macro_apex_score = torch.tensor(macro_apex_score).float()
        micro_action_score = torch.tensor(micro_action_score).float()
        macro_action_score = torch.tensor(macro_action_score).float()

        if self._split == "train":
            return (feature, micro_apex_score, macro_apex_score,
                    micro_action_score, macro_action_score,
                    micro_start_end_label, macro_start_end_label)
        elif self._split == "test":
            return feature, offset, vid_name

    def _has_expression(self, npz_path):
        data = np.load(npz_path)
        labels = data['label']
        micro_action_frame_count = np.sum(labels[:, 3]).item()
        macro_action_frame_count = np.sum(labels[:, 7]).item()
        if micro_action_frame_count + macro_action_frame_count > 0:
            return True
        else:
            return False

    def _has_micro_expression(self, npz_path):
        data = np.load(npz_path)
        labels = data['label']
        micro_action_frame_count = np.sum(labels[:, 3]).item()
        if micro_action_frame_count > 0:
            return True
        else:
            return False

    def _record_feat_path(self):
        feat_dir_root = os.path.join(self._segment_feat_root, self._split)
        expression_feat_file_path_list = []
        nature_feat_file_path_list = []
        if self._split == "train":
            for subject_name in os.listdir(feat_dir_root):
                if self._subject == subject_name:
                    continue
                npz_path_list = glob.glob(
                    os.path.join(feat_dir_root, subject_name, "*.npz"))
                npz_path_list.sort()
                for npz_path in npz_path_list:
                    if self._has_expression(npz_path):
                        expression_feat_file_path_list.append(npz_path)
                    else:
                        nature_feat_file_path_list.append(npz_path)
        elif self._split == "test":
            expression_feat_file_path_list = glob.glob(
                os.path.join(feat_dir_root, self._subject, "*.npz"))
        random.seed(42)
        feat_file_path_list = expression_feat_file_path_list
        shuffle(feat_file_path_list)
        self._feat_file_path_list = feat_file_path_list



if __name__ == "__main__":
    dataset = "cas(me)^2"
    with open("./config.yaml", encoding="UTF-8") as f:
        opt = yaml.safe_load(f)[dataset]
    subject_list = opt['subject_list']
    set_seed(seed=42)
    
    test_dataset = LOSO_DATASET(opt, 'test', subject_list[0])
    test_dataset[0]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8,
        shuffle=False, num_workers=0)

    for (feature, offset, vid_name) in test_loader:
        print(feature.shape)
        print(offset)
        print(vid_name)