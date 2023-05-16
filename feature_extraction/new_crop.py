import os
import glob
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml
import shutil

from tools import FaceDetector


def get_img_count(root_path, dataset):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if not sub_item.is_dir():
            continue
        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue
            if (dataset == "samm_25"
                    and (type_item.name == "032_3"
                         or type_item.name == "032_6")):
                continue
            count += len(glob.glob(os.path.join(str(type_item), "*.jpg")))
    return count


def crop(opt):
    try:
        simpled_root_path = opt["simpled_root_path"]
        cropped_root_path = opt["cropped_root_path"]
        CROPPED_SIZE = opt["CROPPED_SIZE"]
        dataset = opt["dataset"]
    except KeyError:
        print(f"Dataset {dataset} does not need to be cropped")
        print("terminate")
        exit(1)

    sum_count = get_img_count(simpled_root_path, dataset)
    print("img count = ", sum_count)

    if not os.path.exists(simpled_root_path):
        print(f"path {simpled_root_path} is not exist")
        exit(1)

    if not os.path.exists(cropped_root_path):
        os.makedirs(cropped_root_path)

    face_det_model_path = "./checkpoint/retinaface_Resnet50_Final.pth"
    face_detector = FaceDetector(face_det_model_path)

    with tqdm(total=sum_count) as tq:
        for sub_item in Path(simpled_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue

                new_dir_path = os.path.join(
                    cropped_root_path, sub_item.name, type_item.name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                # there will be some problem when crop face from 032_3 032_6.
                # These two directory should be copied to croped directory
                # directly.
                if (dataset == "samm_25"
                    and (type_item.name == "samm_032_3"
                         or type_item.name == "samm_032_6")):
                    shutil.copytree(
                        str(type_item), new_dir_path, dirs_exist_ok=True)
                    continue

                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                if len(img_path_list) > 0:
                    img_path_list.sort()
                    for index, img_path in enumerate(img_path_list):
                        img = cv2.imread(img_path)
                        if index == 0:
                            h, w, c = img.shape
                            face_left, face_top, face_right, face_bottom = \
                                face_detector.cal(img)
                            face_width = (face_right - face_left + 1)
                            face_height = (face_bottom - face_top + 1)
                            vertical_remain = h - face_height
                            horizontal_remain = w - face_width
                            # vertical_remain_top = face_top
                            vertical_remain_bottom = h - face_bottom - 1
                            # horizontal_remain_left = face_left
                            horizontal_remain_right = w - face_right - 1
                            padding_bottom = int(
                                vertical_remain_bottom
                                / vertical_remain
                                * (CROPPED_SIZE - face_height))
                            padding_top = (
                                (CROPPED_SIZE - face_height) - padding_bottom)
                            padding_right = int(
                                horizontal_remain_right
                                / horizontal_remain
                                * (CROPPED_SIZE - face_width))
                            padding_left = (
                                (CROPPED_SIZE - face_width) - padding_right)

                            # clip_left = face_left - padding_left
                            # clip_right = face_right + padding_right
                            # clip_top = face_top - padding_top
                            # clip_bottom = face_bottom + padding_bottom
                            clip_left = face_left 
                            clip_right = face_right 
                            clip_top = face_top 
                            clip_bottom = face_bottom 

                        img = img[clip_top:clip_bottom + 1,
                                  clip_left:clip_right + 1, :]
                        cv2.imwrite(os.path.join(
                                    new_dir_path,
                                    f"img_{str(index+1).zfill(5)}.jpg"), img)
                        tq.update()


if __name__ == "__main__":
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    crop(opt)
