import os
import glob
from pathlib import Path
import random
from random import shuffle
import shutil

import pandas as pd
from sklearn.pipeline import FeatureUnion


def _split(dataset, split_path, split):
    for feature_path in dataset:
        sub_dir_path, _ = os.path.split(feature_path)
        _, sub_dir_name = os.path.split(sub_dir_path)

        new_sub_dir_path = os.path.join(
            split_path, sub_dir_name)
        if os.path.exists(new_sub_dir_path) is not True:
            os.makedirs(new_sub_dir_path)
        shutil.copy(feature_path, new_sub_dir_path)

    df = pd.read_csv("./cross.csv")
    df_list = []
    for feature_path in dataset:
        sub_dir_path, feature_name = os.path.split(feature_path)
        video_name = os.path.splitext(feature_name)[0]
        df_list.append(df[df["video_name"] == video_name])
    new_df = pd.concat(df_list)
    new_df = new_df.sort_values("video_name")
    new_df.to_csv(f"./cross_{split}.csv", index=False)


if __name__ == "__main__":
    random.seed(42)
    feature_root_path = "/home/whcold/Datas/public_micro_dataset/cross/feature"
    feature_split_path = ("/home/whcold/Datas/public_micro_dataset/cross/"
                          "feature_split")
    train_split_path = os.path.join(feature_split_path, "train")
    test_split_path = os.path.join(feature_split_path, "test")
    feature_path_list = []
    for sub_item in Path(feature_root_path).iterdir():
        if sub_item.is_dir() is not True:
            continue
        feature_path_list += glob.glob(os.path.join(str(sub_item), "*.npy"))

    shuffle(feature_path_list)
    train_sample_count = int(len(feature_path_list) * 0.76)
    test_sample_count = len(feature_path_list) - train_sample_count
    train_set = feature_path_list[: train_sample_count]
    test_set = feature_path_list[train_sample_count:]

    if os.path.exists(train_split_path) is not True:
        os.makedirs(train_split_path)
    if os.path.exists(test_split_path) is not True:
        os.makedirs(test_split_path)

    _split(train_set, train_split_path, "train")
    _split(test_set, test_split_path, "test")
