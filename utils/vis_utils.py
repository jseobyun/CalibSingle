import cv2
import numpy as np

def draw_chessboard(img, corners):
    canvas = img.copy()
    num_corners = np.shape(corners)[0]
    for corner in corners:
        cv2.circle(canvas, (int(corner[0]), int(corner[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)

    for idx in range(num_corners-1):
        if idx == 0:
            color = (0, 0, 255)
        else:
            color = (0, 200, 0)

        cv2.line(
            canvas,
            tuple(corners[idx, :].astype(int)),
            tuple(corners[idx+1, :].astype(int)),
            color, 2
        )
    return canvas


def draw_tags(img, tags):
    canvas = img.copy()
    tags = tags.reshape(-1, 4, 2)
    for tag in tags:
        for idx in range(4):
            cv2.line(
                canvas,
                tuple(tag[idx - 1, :].astype(int)),
                tuple(tag[idx, :].astype(int)),
                (0, 200, 0), 2
            )
            cv2.putText(
                canvas,
                str(idx),
                tuple(tag[idx, :].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # cv2.putText(
        #     canvas,
        #     str(tag.tag_id),
        #     (tag[0].astype(int) - 20,
        #      tag[1].astype(int) + 20,),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 0, 255),
        #     2,
        # )
    return canvas