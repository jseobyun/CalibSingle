import cv2
import torch
import numpy as np

class OutlierFilter():
    def __init__(self, args):
        self.args = args
        self.model = torch.load(args.vae_path, map_location=torch.device("cpu"))
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def binarize(self, img):
        # Otsu's thresholding after Gaussian filtering
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img_out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_out = np.float32(img_out) / 255.0
        return img_out

    def generate_patch(self, img_path, pt2ds):
        # load corner crops'
        crop_size = 15
        s = int(crop_size/2)
        num_points = np.shape(pt2ds)[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        crops = []
        for pt_idx in range(num_points):
            pt2d = np.int32(pt2ds[pt_idx])
            crop = img[pt2d[1]-s:pt2d[1]+s+1, pt2d[0]-s: pt2d[0]+s+1]
            crop = self.binarize(crop)
            crop_h, crop_w = np.shape(crop)[:2]
            crops.append(crop.reshape(-1, 1, crop_h, crop_w))
        crops = np.concatenate(crops, axis=0)
        crops = torch.from_numpy(crops)
        if torch.cuda.is_available():
            crops = crops.cuda()
        return crops

    def filter(self, img_paths, all_pt2ds, all_pt3ds, th=0.1):
        img_paths_filtered = []
        all_pt2ds_filtered = []
        all_pt3ds_filtered = []

        for img_path, pt2ds, pt3ds in zip(img_paths, all_pt2ds, all_pt3ds):

            crops = self.generate_patch(img_path, pt2ds) # B 1 H W
            recons, _, _ = self.model(crops, is_training=False)
            loss = torch.sum(((crops - recons) ** 2), dim=(1, 2, 3)).detach().cpu().numpy()/(15**2)
            valid = loss < th
            if np.sum(valid) == 0:
                continue
            img_paths_filtered.append(img_path)
            all_pt2ds_filtered.append(pt2ds[valid])
            all_pt3ds_filtered.append(pt3ds[valid])
            print("before", np.shape(pt2ds)[0])
            print("after", np.shape(pt2ds[valid])[0])
        return img_paths_filtered, all_pt2ds_filtered, all_pt3ds_filtered





