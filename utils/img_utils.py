import os
import cv2


def video2img(video_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break

        img_name = format(count, "03d") + ".jpg"
        cv2.imwrite(os.path.join(save_path, img_name), img)
        count += 1
    cap.release()
    print(f"Total {count} images are saved at {save_path}.")

def get_img_paths(img_dir):
    img_names = sorted(os.listdir(img_dir))
    img_names = [img_name for img_name in img_names if img_name.endswith(".jpg") or img_name.endswith(".png")]
    img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
    return img_paths

if __name__ == "__main__":
    root = "./data"

    # video2img(os.path.join(root, "chessboard_flipped.mp4"), os.path.join(root, "chessboard_flipped"))



