import math
import numpy as np
import pandas as pd
import cv2

import torch
from SAN.san_api import SanLandmarkDetector
from retinaface.api import Facedetecor as RetinaFaceDetector


def imshow_for_test(windowname, img, face_boundarys=None, landmarks=None):
    if face_boundarys is not None:
        for face_boundary in face_boundarys:
            cv2.rectangle(img, (face_boundary[0], face_boundary[1]),
                          (face_boundary[2], face_boundary[3]), (0, 0, 255), 1)
    if landmarks is not None:
        for point in landmarks:
            cv2.circle(img, (point[0].item(), point[1].item()),
                       1, (0, 0, 255), 4)
    cv2.putText(img, windowname, (15, 15),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    cv2.imshow(windowname, img)
    cv2.waitKey(0)


class LandmarkDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = SanLandmarkDetector(model_path, device)

    def cal(self, img, offset=None, face_box=None):
        if face_box is None:
            face_box = (0, 0, img.shape[1], img.shape[0])
        locs, _ = self.det.detect(img, face_box)
        x_list = [
            loc[0] if offset is None else loc[0] - offset[0] for loc in locs]
        y_list = [
            loc[1] if offset is None else loc[1] - offset[1] for loc in locs]
        return x_list, y_list


class FaceDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = RetinaFaceDetector(model_path, device)

    def cal(self, img):
        left, top, right, bottom = self.det.get_face_box(img)
        return left, top, right, bottom


def get_top_optical_flows(optflows, percent):
    assert type(optflows) == np.ndarray, "optflows must be numpy ndarray"
    tmp_optflows = optflows
    if optflows.ndim == 3 and optflows.shape[-1] == 2:
        tmp_optflows = optflows.reshape(-1, 2)
    elif optflows.ndim == 2 and optflows.shape[-1] == 2:
        tmp_optflows = optflows
    else:
        raise "shape of optflows is invalid"

    length = len(optflows)
    top_n = int(length * percent)
    new_indices = np.argsort(np.linalg.norm(tmp_optflows, axis=-1))
    ret_optflows = tmp_optflows[new_indices][length - top_n:]
    return ret_optflows


def get_rectangle_roi_boundary(indices, landmarks,
                               horizontal_bound=0, vertical_bound=0):
    """ calculate a boundary of a roi that consists of a bunch of landmarks

    Args:
        indices: indices of landmarks, must be tuple, list of numpy.dnarray
    Returns:
        left_bound: left boundary of the rectangle roi
        top_bound: top boundary of the rectangle roi
        right_bound: right boundary of the rectangle roi
        bottom_bound: bottom boundary of the rectangle roi
    """
    assert type(horizontal_bound) == int, "horizontal_bound must be integer"
    assert type(vertical_bound) == int, "vertical_bound must be integer"
    if type(indices) == tuple or type(indices) == list:
        indices = np.array(indices)
    elif type(indices) == np.ndarray:
        pass
    else:
        raise "type of indices is incorrect"

    roi_landmarks = landmarks[indices]
    left_bound, top_bound = np.min(roi_landmarks, axis=0)
    right_bound, bottom_bound = np.max(roi_landmarks, axis=0)
    return left_bound-horizontal_bound, top_bound-vertical_bound, \
        right_bound+horizontal_bound, bottom_bound+vertical_bound


def get_rois(mat, landmarks, indices, horizontal_bound=3, vertical_bound=3):
    """ get rois with indeices of landmarks

    Args:
        mat: a rgb image or flow image
        landmarks: landmarks of face region
        indeices: indeices of landmarks
        horizontal_bound:
        vertical_bound:
    Returns:
        a ndarray of roi mat
    """

    if type(indices) == tuple or type(indices) == list:
        indices = np.array(indices)
    elif type(indices) == np.ndarray:
        pass
    else:
        raise "type of indices is incorrect"

    assert type(landmarks) == np.ndarray, "landmarks should be numpy.ndarray"

    roi_list = []
    for landmark in landmarks[indices]:
        x = landmark[0].item()
        y = landmark[1].item()
        roi_list.append(mat[y - vertical_bound: y + vertical_bound + 1,
                            x - horizontal_bound: x + horizontal_bound + 1, :])
    return np.stack(roi_list, axis=0)


def optflow_normalize(flow):
    """ normalize optical flows

    Args:
        flow: np.ndarry, shape of flow should be (-1, 2)

    Returns:
        a np.ndarray, the shape of return is (2,)
    """

    assert flow.dtype == np.float32, (
        "element type of optflow should be float32")

    delta = 0.000001
    sum_flow = np.sum(flow, axis=0)
    flow_one = sum_flow / (np.linalg.norm(sum_flow) + delta)
    average_module = np.sum(np.linalg.norm(flow, axis=1)) / flow.shape[0]
    feature = flow_one * average_module
    return feature


def get_main_direction_flow(array_flow, direction_region):
    """get all the flow vectors that are main directional in a region of flow

    Args:
        array_flow: a ndarray of flows
    Returns:
        a ndarray of flows that are main directional in a region of flow
    """

    array_flow = array_flow.reshape(-1, 2)
    _, angs = cv2.cartToPolar(array_flow[..., 0], array_flow[..., 1])
    direction_flows = [[] for i in range(len(direction_region))]

    for i, ang in enumerate(angs):
        for index, direction in enumerate(direction_region):
            if len(direction) == 2:
                if ang >= direction[0] and ang < direction[1]:
                    direction_flows[index].append(array_flow[i])
                    break
            elif len(direction) == 4:
                if (ang >= direction[0]
                        or (ang >= direction[2] and ang < direction[3])):
                    direction_flows[index].append(array_flow[i])
                    break

    max_count_index = np.argmax(
        np.array([len(x) for x in direction_flows])).item()

    return np.stack(direction_flows[max_count_index], axis=0)


def cal_global_optflow_vector(flows, landmarks):
    """calculates optical flow vector of nose region

    calculates array of optical flows of nose region as the global optical flow
    to indicate head motion, and then calculates the normalized vector of the
    array.

    Args:
        flows: flows of a image
        landmarks: landmarks of the face region
    Returns:
        global optical flow vector.
    """

    def _cal_partial_opt_flow(indices, horizontal_bound, vertical_bound):

        (nose_roi_left, nose_roi_top, nose_roi_right,
            nose_roi_bottom) = get_rectangle_roi_boundary(
                indices, landmarks,
                horizontal_bound, vertical_bound)

        flow_nose_roi = flows[
            nose_roi_top: nose_roi_bottom + 1,
            nose_roi_left: nose_roi_right + 1]

        flow_nose_roi = flow_nose_roi.reshape(-1, 2)
        return flow_nose_roi

    LEFT_EYE_CONER_INDEX = 39
    RIGHT_EYE_CONER_INDEX = 42
    left_eye_coner = landmarks[LEFT_EYE_CONER_INDEX]
    right_eye_coner = landmarks[RIGHT_EYE_CONER_INDEX]
    length_between_coners = (right_eye_coner[0] - left_eye_coner[0]) / 2

    flow_nose_roi_list = []
    flow_nose_roi_list.append(
        _cal_partial_opt_flow(
            np.arange(29, 30 + 1),
            horizontal_bound=int(length_between_coners * 0.35),
            vertical_bound=int(length_between_coners * 0.35)))
    flow_nose_roi = np.stack(flow_nose_roi_list).reshape(-1, 2)
    flow_nose_roi = get_main_direction_flow(
        flow_nose_roi,
        direction_region=[
            (1 * math.pi / 4, 3 * math.pi / 4),
            (3 * math.pi / 4, 5 * math.pi / 4),
            (5 * math.pi / 4, 7 * math.pi / 4),
            (7 * math.pi / 4, 8 * math.pi / 4, 0, 1 * math.pi / 4),
        ])
    flow_nose_roi = get_top_optical_flows(flow_nose_roi, percent=0.88)
    glob_flow_vector = optflow_normalize(flow_nose_roi)
    return glob_flow_vector


def calculate_roi_freature_list(flow, landmarks, radius):
    assert flow.dtype == np.float32, (
        "element type of optflow should be float32")
    assert np.max(flow) <= 1, "max value shoued be less than 1"

    ior_flows = get_rois(
        flow, landmarks,
        indices=[
            18, 19, 20,     # left eyebrow
            23, 24, 25,     # right eyebrow
            28, 30,         # nose
            48, 51, 54, 57  # mouse
        ],
        horizontal_bound=radius,
        vertical_bound=radius
    )

    global_optflow_vector = cal_global_optflow_vector(flow, landmarks)

    ior_flows_adjust = ior_flows - global_optflow_vector
    ior_feature_list = []   # feature in face
    for ior_flow in ior_flows_adjust:
        ior_main_direction_flow = get_main_direction_flow(
            ior_flow,
            direction_region=[
                (1 * math.pi / 6, 5 * math.pi / 6),
                (5 * math.pi / 6, 7 * math.pi / 6),
                (7 * math.pi / 6, 11 * math.pi / 6),
                (11 * math.pi / 6, 12 * math.pi / 6, 0, 1 * math.pi / 6),
            ])
        ior_main_direction_flow = get_top_optical_flows(
            ior_main_direction_flow, percent=0.6)
        ior_feature = optflow_normalize(ior_main_direction_flow)
        ior_feature_list.append(ior_feature)
    return np.stack(ior_feature_list, axis=0)


def conver_flow_to_gbr(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def get_micro_expression_average_len(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["type_idx"] == 2]
    df = df[df["end_frame"] != 0]
    array_start_frame = df.start_frame.values
    array_end_frame = df.end_frame.values
    array_me_len = array_end_frame - array_start_frame + 1
    average_len = np.mean(array_me_len).item()
    return average_len


def get_macro_expression_average_len(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["type_idx"] == 1]
    df = df[df["end_frame"] != 0]
    array_start_frame = df.start_frame.values
    array_end_frame = df.end_frame.values
    array_me_len = array_end_frame - array_start_frame + 1
    average_len = np.mean(array_me_len).item()
    return average_len


if __name__ == "__main__":
    a = get_micro_expression_average_len("./samm_new_25.csv")
    pass
