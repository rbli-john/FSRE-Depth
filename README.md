# Annotation for Instance Segmentation
Use dectectron2 pretrained weight ```COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml``` to generate panoptic(instance) ground truth.
## Usage
```
python generate_ground_truth.py --data_path [Root Dir of Kitti]
```
Generate ```annotation.json```. Takes about 40 mins.
## Format
```
{
"annotations": [
  "file_name": image_file_name
  "image_id": unique_id_for_each_image
  "segments_info": [segment_info]
  
  segment_info: {
    "area":
    "bbox":
    "category_id":
    "is_crowd":
  }
]
"categories": [
  "id"
  "isthing"
  "name"
]
"images": [
  "file_name"
  "height"
  "width"
  "id"
]
}
```
## Files
- ```demo.png```: panoptic segmentation visualization result for ```image.png```. 
## Requirement
```
detectron2
torch
torchvision
```
