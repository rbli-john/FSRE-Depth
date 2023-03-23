import argparse
import os

import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2
import numpy as np

from datasets.kitti_utils import generate_depth_map
from utils import readlines


def export_gt_depths_kitti():
    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        default='eigen',
                        choices=["eigen", "eigen_benchmark", "predictions"]) # "predictions" for small batch of self selected images
    opt = parser.parse_args()

    if opt.split == "predictions":
        split_folder = os.path.join(os.path.dirname(__file__), "predictions", "image")
    else:
        split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for idx, line in enumerate(lines):

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        elif opt.split == "predictions":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
            print("gt_depth:", gt_depth.shape) # 375, 1242

            # heat map
            disp_resized = cv2.resize(gt_depth, (1216, 352))
            # # depth = STEREO_SCALE_FACTOR * 5.2229753 / disp_resized
            # depth = 32.779243 / disp_resized
            depth = disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            # MIN_DEPTH = 1e-3
            # MAX_DEPTH = 80
            # mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH) # 446574 19176, 96%
            # print(np.count_nonzero(np.logical_not(mask)), np.count_nonzero(mask))
            save_path = os.path.join("predictions", "ground_truth", "{:010d}.png".format(idx))
            plt.clf()
            # plt.figure(figsize = (17, 5), dpi=100)
            plt.imshow(depth.astype(np.float32), cmap=plt.get_cmap('plasma'), interpolation='nearest')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
