import os
import cv2
import json
import numpy as np
from pupil_apriltags import Detector

### for Window
# os.add_dll_directory('C:/Users/whdtj/anaconda3/envs/calibsingle/Lib/site-packages/pupil_apriltags.libs')

at_detector = Detector(
   families="tagStandard41h12",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def get_tag3d(cell_size, ridx, cidx, v_dist, h_dist):
    tag3d = np.array([[0, 0, 0],
                      [5, 0, 0],
                      [5, 5, 0],
                      [0, 5, 0]], dtype=np.float32) * cell_size
    tag3d[:, 0] += cidx * (9*cell_size) + cidx*h_dist
    tag3d[:, 1] -= ridx * (9*cell_size) + ridx*v_dist

    return tag3d

def create_tagboard(board):
    num_row, num_col = board["num_row"], board["num_col"]
    cell_size = board["cell_size"]
    h_dist = board["h_dist"]
    v_dist = board["v_dist"]
    start_id = board["start_id"]
    tags = {}
    for ridx in range(num_row):
        for cidx in range(num_col):
            target_id = start_id + (ridx*num_col) + cidx
            tags[target_id] = get_tag3d(cell_size, ridx, cidx, v_dist, h_dist)
    board["tags"] = tags
    return board

def create_chessboard(board):
    num_row, num_col = board["num_row"], board["num_col"]
    cell_size = board["cell_size"]

    corner3ds = []
    for ridx in range(num_row):
        for cidx in range(num_col):
            target_id = (ridx*num_col) + cidx
            corner3ds.append(np.array([cidx*cell_size, -ridx*cell_size, 0], dtype=np.float32).reshape(-1,3))
    corner3ds = np.concatenate(corner3ds, axis=0)
    board["corner3ds"] = corner3ds
    return board

def load_board(board_path):
    if not os.path.exists(board_path):
        raise Exception(f"{board_path} does not exist.")
    with open(board_path, 'r') as json_file:
        board_metainfo = json.load(json_file)

    board_type = board_metainfo["type"]
    if board_type.lower() == "tagboard":
        board_dict = create_tagboard(board_metainfo)
    elif board_type.lower() == "chessboard":
        board_dict = create_chessboard(board_metainfo)
    else:
        raise NotImplemented(f"{board_type} is not supported yet.")
    return board_dict


def detect_apriltag(img, board, refine=True):
    if len(np.shape(img)) == 3 and np.shape(img)[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    detections = at_detector.detect(img_gray)
    tags = {}
    for detection in detections:
        corners = np.array(detection.corners).reshape(-1,2).astype(np.float32)

        id = detection.tag_id
        if refine:
            # only float32 supported
            corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        if id in list(board["tags"].keys()):
            tags[id] = corners

    if len(list(tags.keys())) == 0:
        return None
    else:
        return tags

def detect_chessboard(img, board, refine=True):
    if len(np.shape(img)) == 3 and np.shape(img)[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    num_row, num_col = board["num_row"], board["num_col"]
    ret, corners = cv2.findChessboardCorners(img_gray, (num_col, num_row), None)
    if refine:
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

    if not ret:
        return None
    else:
        return corners
