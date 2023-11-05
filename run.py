import os
import cv2
import argparse
import numpy as np
from detector import Detector
from calibrator import Calibrator

from utils.img_utils import get_img_paths
from utils.vis_utils import draw_tags, draw_chessboard

def parse_config():
    parser = argparse.ArgumentParser(description="Single camera calibration")
    ## necessary
    parser.add_argument("--data_dir", type=str, default="/home/jseob/Desktop/yjs/codes/Calibration/data/20230626_alone2nd_calib/calibration_image/intrinsic/rgb3", help="image directory path")
    #parser.add_argument("--data_dir", type=str, default="./data/chessboard", help="image directory path")
    parser.add_argument("--board", type=str, default="./board/tagboard.json", help="calibration board path")
    parser.add_argument("--camera_model", type=str, default="pinhole", choices=["pinhole", "fisheye"], help="camera model")
    parser.add_argument("--init_model", type=str, default="cholesky", choices=["cholesky", "zhang", "burger"])
    parser.add_argument("--is_wide_angle", default=False, action="store_true", help="fov over 120 or not")

    ## optional
    parser.add_argument("--num_images", type=int, default=20, help="the maximum number of images used")
    parser.add_argument("--do_filtering", default=False, action="store_true", help="do point filtering based on VAE")
    parser.add_argument("--vae_path", default="./filter/vae_model.pt")
    parser.add_argument("--vis", default=True, action="store_true", help="do visualization")
    parser.add_argument("--vis_dir", type=str, default="./data/tagboard_vis", help="visualization directory path when vis==True")

    args = parser.parse_args()
    if args.vis:
        os.makedirs(args.vis_dir, exist_ok=True)

    return args

if __name__ == "__main__":
    args = parse_config()

    detector = Detector(args)
    img_paths, all_pt2ds, all_pt3ds = detector.detect_all(args.data_dir)
    if args.do_filtering and os.path.exists(args.vae_path) :
        from filter import OutlierFilter
        from filter.vae_model import OutlierVAE, View # do not remove it.
        outlier_filter = OutlierFilter(args)
        img_paths, all_pt2ds, all_pt3ds = outlier_filter.filter(img_paths, all_pt2ds, all_pt3ds, loss_th=0.4)

    ## vis
    # img_paths = get_img_paths(args.data_dir)
    # for img_path, pt2ds in zip(img_paths, all_pt2ds):
    #     img = cv2.imread(img_path)
    #     canvas = draw_tags(img, pt2ds)
    #     #canvas = draw_chessboard(img, pt2ds)
    #
    #     cv2.imshow("vis", cv2.resize(canvas, dsize=(0,0), fx=0.5, fy=0.5))
    #     cv2.waitKey(0)
    # ##

    calibrator = Calibrator(args)
    calibrator.register(img_paths, all_pt2ds, all_pt3ds)

    ### from scratch version
    calibrator.calibrate(use_opencv=False)
    calibrator.print_results()
    if args.vis:
        calibrator.visualize()

    ### from opencv
    calibrator.calibrate(use_opencv=True)
    calibrator.print_results()
    if args.vis:
        calibrator.visualize()


    

    
