import os
import cv2
import argparse
import numpy as np
from detector import Detector
from calibrator import Calibrator

from utils.img_utils import get_img_paths

def parse_config():
    parser = argparse.ArgumentParser(description="Single camera calibration")
    ## necessary
    parser.add_argument("--data_dir", type=str, default="./data/chessboard", help="image directory path")
    parser.add_argument("--board", type=str, default="./board/chessboard.json", help="calibration board path")
    parser.add_argument("--camera_model", type=str, default="pinhole", choices=["pinhole", "fisheye"], help="camera model")

    ## optional
    parser.add_argument("--num_images", type=int, default=20, help="the maximum number of images used")
    parser.add_argument("--do_filtering", default=True, action="store_true", help="do point filtering based on VAE")
    parser.add_argument("--vae_path", default="./filter/vae_model.pt")
    parser.add_argument("--vis", default=False, action="store_true", help="do visualization")
    parser.add_argument("--vis_dir", type=str, default="", help="visualization directory path when vis==True")

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
        img_paths, all_pt2ds, all_pt3ds = outlier_filter.filter(img_paths, all_pt2ds, all_pt3ds)

    ### vis
    img_paths = get_img_paths(args.data_dir)
    for img_path in img_paths:
        img = cv2.imread(img_path)

    '''    
    calibrator = Calibrator(args)    
            
    frames = detector.detect()
    if args.do_filtering:
        filter = load_filter(args)
        frames = filter.filter(frames)
            
    calibrator.register(frames) 
    
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
    
    '''