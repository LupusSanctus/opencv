from dataclasses import dataclass
import os


@dataclass
class TestConfig:
    frame_size: int = 224
    imgs_class_dir: str = "./ILSVRC2012_img_val"
    imgs_segm_dir_root: str = "./VOC2012"
    imgs_segm_dir: str = os.path.join(imgs_segm_dir_root, "JPEGImages")
    imgs_segm_gt_dir: str = os.path.join(imgs_segm_dir_root, "SegmentationClass")
    segm_val_file: str = os.path.join(imgs_segm_dir_root, "ImageSets/Segmentation/val.txt")
    batch_size: int = 100
    # location of image-class matching
    img_cls_file: str = "./val.txt"
    log: str = "./log.txt"
    bgr_to_rgb: bool = True
