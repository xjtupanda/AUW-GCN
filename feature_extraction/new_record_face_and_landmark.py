import os
import glob
import csv
from pathlib import Path

import yaml
import cv2
from tqdm import tqdm
from tools import LandmarkDetector, FaceDetector


def get_img_count(cropped_root_path):
    count = 0
    for sub_item in Path(cropped_root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    count += len(
                        glob.glob(os.path.join(str(type_item), "*.jpg")))
    return count


def record_csv(csv_path, rows):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(rows)


def show_img(img, face, landmarks):
    for i in range(len(landmarks) // 2):
        cv2.circle(img, (landmarks[i], landmarks[i + 68]), 1, (0, 0, 255), 4)
    cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 1)
    cv2.imshow("landmark", img)
    cv2.waitKey(0)


def record_face_and_landmarks(opt):
    cropped_root_path = opt["cropped_root_path"]

    if not os.path.exists(cropped_root_path):
        print(f"path {cropped_root_path} is not exist")
        exit(1)

    sum_count = get_img_count(cropped_root_path)
    print("img count = ", sum_count)

    face_det_model_path = "./checkpoint/retinaface_Resnet50_Final.pth"
    face_detector = FaceDetector(face_det_model_path)
    landmark_model_path = './checkpoint/san_checkpoint_49.pth.tar'
    landmark_detector = LandmarkDetector(landmark_model_path)

    with tqdm(total=sum_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                if len(img_path_list) > 0:
                    img_path_list.sort()
                    rows_face = []
                    rows_landmark = []
                    csv_face_path = os.path.join(str(type_item), "face.csv")
                    csv_landmark_path = os.path.join(
                        str(type_item), "landmarks.csv")
                    for index, img_path in enumerate(img_path_list):
                        img = cv2.imread(img_path)
                        try:
                            left, top, right, bottom = face_detector.cal(img)
                            x_list, y_list = landmark_detector.cal(
                                img, face_box=(left, top, right, bottom))
                        except Exception:
                            print(f"subject: {sub_item.name}, "
                                  "em_type: {type_item.name}, index: {index}")
                            break
                        # print(f"face_width: {right-left+1}")
                        # print(f"face_height: {bottom-top+1}")
                        # print(f"image_width: {img.shape[1]}")
                        # print(f"image_width: {img.shape[0]}")
                        # show_img(img, (left, top, right, bottom),
                        #          x_list + y_list)

                        rows_face.append((left, top, right, bottom))
                        rows_landmark.append(x_list + y_list)
                        tq.update()
                    if len(rows_face) == len(img_path_list):
                        record_csv(csv_face_path, rows_face)
                        record_csv(csv_landmark_path, rows_landmark)


if __name__ == "__main__":
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    record_face_and_landmarks(opt)
