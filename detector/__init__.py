from utils.board_utils import *
from utils.img_utils import get_img_paths
class Detector():
    def __init__(self, args):
        self.args = args
        self.board_metainfo = load_board(args.board)
        self.board_type = self.board_metainfo["type"]
        if self.board_type == "chessboard":
            self.board = create_chessboard(self.board_metainfo)
        elif self.board_type == "tagboard":
            self.board = create_tagboard(self.board_metainfo)


    def detect_all(self, img_dir):
        img_paths = get_img_paths(img_dir)[:self.args.num_images]
        all_img_paths = []
        all_pt2ds = []
        all_pt3ds = []
        for img_path in img_paths:
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            pt2ds, pt3ds = self.detect(img_gray)
            if pt2ds is None or pt3ds is None:
                continue
            all_img_paths.append(img_path)
            all_pt2ds.append(pt2ds)
            all_pt3ds.append(pt3ds)
        return all_img_paths, all_pt2ds, all_pt3ds

    def detect(self, img):
        if self.board_type == "tagboard":
            tags_dict = detect_apriltag(img, self.board, refine=True)
            if tags_dict is None:
                return None, None
            tag2ds = []
            tag3ds = []
            for id in list(tags_dict.keys()):
                tag2d = tags_dict[id].reshape(4,2)
                tag3d = self.board["tags"][id].reshape(4,3)
                tag2ds.append(tag2d)
                tag3ds.append(tag3d)
            tag2ds = np.concatenate(tag2ds, axis=0)
            tag3ds = np.concatenate(tag3ds, axis=0)
            return tag2ds, tag3ds

        elif self.board_type =="chessboard":
            corner2ds = detect_chessboard(img, self.board, refine=True)
            if corner2ds is None:
                return None, None
            corner2ds = corner2ds.squeeze(axis=1)
            corner3ds = self.board["corner3ds"]
            return corner2ds, corner3ds
        else:
            raise NotImplemented()








