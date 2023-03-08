import argparse
import json
from tqdm import tqdm
from glob import glob
import os
import torch, torchvision
# Some basic setup:
# Setup detectron2 logger
# from skimage import measure                        # (pip install scikit-image)
# from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# from panopticapi.utils import id2rgb

category_names = set()

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask = sub_mask.astype(int)
    print(sub_mask)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

def generate_gt_per_image(predictor, cfg, image_path, data_path, image_id):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (640, 192), interpolation= cv2.INTER_AREA) # same as FSRE
    H, W, C = im.shape

    output = predictor(im)
    predictions, segmentInfo = output["panoptic_seg"]
    instances = output["instances"]
    # rgb = id2rgb(predictions.cpu().numpy())
    # cv2.imwrite("predictions.png",rgb)
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)

    # Uncomment to filter out specific segments
    # out = v.draw_panoptic_seg_predictions(predictions.to("cpu"), list(filter(lambda x: x['category_id'] == 17, segmentInfo)), area_threshold=.1)
    # out = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo, area_threshold=.1)
    # print("Writing demo image...")
    # cv2.imwrite("demo.png", out.get_image()[:, :, ::-1])



    file_name = image_path.replace(data_path, '')
    annotation_per_file = {"file_name": file_name, "image_id": image_id, "segments_info": []}
    image_per_file = {"file_name": file_name, "height": H, "width": W, "id": image_id}
    category_datas = []

    predictions = predictions.cpu().numpy()
    fields = getattr(instances, '_fields')
    bboxs = getattr(fields['pred_boxes'], 'tensor').cpu().numpy()
    scores = fields['scores'].cpu().numpy()
    pred_classes = fields['pred_classes'].cpu().numpy()
    pred_masks = fields['pred_masks'].cpu().numpy()
    num_instances = len(bboxs)
    segmentInfo_dict = {}
    
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    thing_classes = metadata.thing_classes
    stuff_classes = metadata.stuff_classes

    for element in segmentInfo:
        segmentInfo_dict[int(element['id'])] = element
        if element['isthing'] and thing_classes[element['category_id']] not in category_names:
            category_data = {"id": element['category_id'], "isthing": True, "name": thing_classes[element['category_id']]}
            category_names.add(thing_classes[element['category_id']])
            category_datas.append(category_data)
        
        elif (not element['isthing']) and stuff_classes[element['category_id']] not in category_names:
            category_data = {"id": element['category_id'], "isthing": False, "name": stuff_classes[element['category_id']]}
            category_names.add(stuff_classes[element['category_id']])
            category_datas.append(category_data)


    for i in range(num_instances):
        mask = instances.pred_masks[i].cpu().numpy()
        # segmentation = create_sub_mask_annotation(mask)
        _id = np.bincount(predictions[mask]).argmax()
        # if _id not in segmetInfo_dict:
        #     continue
        # seg_element = segmentInfo_dict[_id]
        area = int(np.count_nonzero(mask))
        bbox = bboxs[0]
        bbox = [float(b) for b in bbox]
        category_id = int(pred_classes[i])
        # if category_id != seg_element['category_id']:
        #     # print(i, category_id, seg_element['category_id'])
        #     continue
        # assert category_id == seg_element['category_id']
        annotation_per_file["segments_info"].append({"area": area, "bbox": bbox, "category_id": category_id, "iscrowd": 0}) # "segmentation": segmentation

    return annotation_per_file, image_per_file, category_datas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/chuanqi_chen/cs231-final-backup/Kitti/")
    args = parser.parse_args()
    sub_folders = ['2011_09_26', '2011_09_29', '2011_10_03', '2011_09_28', '2011_09_30']
    sub_folder = sub_folders[0]
    image_paths1 = glob(os.path.join(args.data_path, sub_folder, "*_sync", 'image_02', 'data', "*.png"))
    image_paths2 = glob(os.path.join(args.data_path, sub_folder, "*_sync", 'image_03', 'data', "*.png"))
    image_paths = image_paths1 + image_paths2

    
    cfg = get_cfg()
    # Panoptic Segmentation
    # Ref: https://youtu.be/Pb3opEFP94U
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    


    gt = {"annotations": [], "images": [], "categories": []}

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
       annotation_per_file, image_per_file, category_datas = generate_gt_per_image(predictor, cfg, image_path, args.data_path, i)
       gt["annotations"].append(annotation_per_file)
       gt["images"].append(image_per_file)
       gt["categories"].extend(category_datas)

    with open("annotation.json", "w") as outfile:
        json.dump(gt, outfile, indent=2)















