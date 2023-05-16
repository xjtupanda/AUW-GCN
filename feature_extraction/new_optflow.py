import os
import glob
from pathlib import Path
from tqdm import tqdm
from tools import get_micro_expression_average_len

import yaml


def get_dir_count(root_path):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    if len(glob.glob(
                            os.path.join(str(type_item), "*.jpg"))) > 0:
                        count += 1
    return count


def optflow(opt):
    cropped_root_path = opt["cropped_root_path"]
    optflow_root_path = opt["optflow_root_path"]
    #anno_csv_path = opt["anno_csv_path"]

    if not os.path.exists(cropped_root_path):
        print(f"path {cropped_root_path} is not exist")
        exit(1)
    if not os.path.exists(optflow_root_path):
        os.makedirs(optflow_root_path)

    dir_count = get_dir_count(cropped_root_path)
    print("flow count = ", dir_count)

    opt_step = 1 # int(get_micro_expression_average_len(anno_csv_path) // 2)

    with tqdm(total=dir_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if sub_item.is_dir():
                new_sub_dir_path = os.path.join(
                    optflow_root_path, sub_item.name)
                if not os.path.exists(new_sub_dir_path):
                    os.makedirs(new_sub_dir_path)
                for type_item in sub_item.iterdir():
                    if type_item.is_dir():
                        cmd = (f'denseflow "{str(type_item)}" -b=10 -a=tvl1 '
                               f'-s={opt_step} -if -o="{new_sub_dir_path}"')
                        os.system(cmd)
                        tq.update()


if __name__ == "__main__":
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    optflow(opt)
