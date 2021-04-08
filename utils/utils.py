import numpy as np
import matplotlib.pyplot as plt
import cv2

N_KEYPOINTS = 21
IMAGE_SIZE = 224

COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}


def projectPoints(xyz, K):
    """
    Project 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def show_sample(image, keypoints, figsize=[5, 5]):
    """
    Shows image with annotations
    Inputs:
    - image: PIL image
    - keypoints: np.array of size (21,2)
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="k", alpha=0.5)
    for finger, params in COLORMAP.items():
        plt.plot(
            keypoints[params["ids"], 0], keypoints[params["ids"], 1], params["color"]
        )
    #plt.axis("off")
    plt.show()
    
    
def vector_to_heatmaps(keypoints):
    heatmaps = np.zeros([N_KEYPOINTS, IMAGE_SIZE, IMAGE_SIZE])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        if (0 <= x < IMAGE_SIZE) and (0 <= y < IMAGE_SIZE):
            heatmaps[k, y, x] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps):
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred