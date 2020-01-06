"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
import numpy
from PIL import Image, ImageDraw

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

from pathlib import Path 
import skimage.draw

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco" 

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    number_image_train = 800
    number_image_val = 200

    STEPS_PER_EPOCH = int(number_image_train/IMAGES_PER_GPU)
    VALIDATION_STEPS = int(number_image_val/IMAGES_PER_GPU)

    useDefault = False

    if useDefault:
        # BACKBONE = "resnet50"
        BACKBONE = "resnet101"
        
        # The strides of each layer of the FPN Pyramid. These values
        # are based on a Resnet101 backbone.
        BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        # BACKBONE_STRIDES = [2, 4, 8, 16, 32]

        # Size of the fully-connected layers in the classification graph
        FPN_CLASSIF_FC_LAYERS_SIZE = 1024
        # FPN_CLASSIF_FC_LAYERS_SIZE = 512

        # Size of the top-down layers used to build the feature pyramid
        TOP_DOWN_PYRAMID_SIZE = 256
        # TOP_DOWN_PYRAMID_SIZE = 128

        # Length of square anchor side in pixels
        RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
        

        # Ratios of anchors at each cell (width/height)
        # A value of 1 represents a square anchor, and 0.5 is a wide anchor
        RPN_ANCHOR_RATIOS = [0.5, 1, 2]

        # Anchor stride
        # If 1 then anchors are created for each cell in the backbone feature map.
        # If 2, then anchors are created for every other cell, and so on.
        RPN_ANCHOR_STRIDE = 1

        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.7

        # How many anchors per image to use for RPN training
        RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        # RPN_TRAIN_ANCHORS_PER_IMAGE = 128
    else:
        n = 4
        BACKBONE = "resnet50"
        BACKBONE_STRIDES = [int(4/n), int(8/n), int(16/n), int(32/n), int(64/n)]
        FPN_CLASSIF_FC_LAYERS_SIZE = int(512/n)
        TOP_DOWN_PYRAMID_SIZE = int(256/n)
        RPN_ANCHOR_SCALES = (int(32/n), int(64/n), int(128/n), int(256/n), int(512/n))
        RPN_TRAIN_ANCHORS_PER_IMAGE = int(256/n)

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        class_ids = [0, 1]

        # All images or a subset?
        # if class_ids:
        #     image_ids = []
        #     for id in class_ids:
        #         image_ids.extend(list(coco.getImgIds(catIds=[id])))
        #     # Remove duplicates
        #     image_ids = list(set(image_ids))
        # else:
        #     # All images
        #     image_ids = list(coco.imgs.keys())

        # Add classes
        self.add_class("coco", 0, "Loai 1")
        self.add_class("coco", 1, "Loai 2")
        # for i in class_ids:
        #     self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        with open("{}/{}.txt".format(dataset_dir, subset)) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content] 

        for line in content:
            data = line.split(",")
            image_path = os.path.join(image_dir, data[0])
            
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            annotations = data[1:-1]
            half = int(len(annotations)/2)

            all_points_x = [int(value) for value in annotations[:half]]
            # print(annotations, all_points_x)
            region  = {
                "all_points_x": all_points_x,
                "all_points_y": [int(value) for value in annotations[half:]],
                "class": data[-1]
            }
            # print(region)
            polygons = [region]

            self.add_image(
                "coco", image_id=int(data[0].replace(".png", "")),
                path=image_path,
                width=width,
                height=height,
                polygons=polygons
                # annotations=coco.loadAnns(coco.getAnnIds(
                #     imgIds=[i], catIds=class_ids, iscrowd=None))
                    )


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        # instance_masks = []
        # class_ids = []
        # annotations = self.image_info[image_id]["annotations"]
        # # Build mask of shape [height, width, instance_count] and list
        # # of class IDs that correspond to each channel of the mask.
        # for annotation in annotations:
        #     class_id = self.map_source_class_id(
        #         "coco.{}".format(annotation['category_id']))
        #     if class_id:
        #         m = self.annToMask(annotation, image_info["height"],
        #                            image_info["width"])
        #         # Some objects are so small that they're less than 1 pixel area
        #         # and end up rounded out. Skip those objects.
        #         if m.max() < 1:
        #             continue
        #         # Is it a crowd? If so, use a negative class ID.
        #         if annotation['iscrowd']:
        #             # Use negative class ID for crowds
        #             class_id *= -1
        #             # For crowd masks, annToMask() sometimes returns a mask
        #             # smaller than the given dimensions. If so, resize it.
        #             if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
        #                 m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
        #         instance_masks.append(m)
        #         class_ids.append(class_id)

        # # Pack instance masks into an array
        # if class_ids:
        #     mask = np.stack(instance_masks, axis=2).astype(np.bool)
        #     class_ids = np.array(class_ids, dtype=np.int32)
        #     return mask, class_ids
        # else:
        #     # Call super class to return an empty mask
        #     return super(CocoDataset, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        # instance_masks = []
        class_ids = []

        for i, p in enumerate(info["polygons"]):
            class_id = int(p["class"])
            # Get indexes of pixels inside the polygon and set them to 1
            # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # all_points_y = p['all_points_y']
            # all_points_x = p['all_points_x']
            # img = Image.new('L', (info["width"], info["height"]), 0)
            # polygon = list(zip(all_points_x, all_points_y))
            # print(polygon)
            # ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            # mask = numpy.array(img)
            # mask[rr, cc, i] = 1
            # instance_masks.append(mask)
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            class_ids.append(class_id)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # mask = np.stack(instance_masks, axis=2).astype(np.bool)
        # mask = mask.astype(np.bool)

        # _idx = np.sum(mask, axis=(0, 1)) > 0
        # mask = mask[:, :, _idx]
        # class_ids = class_ids[_idx]

        return mask.astype(np.bool), np.asarray(class_ids,  dtype=np.int32)
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            # return "http://cocodataset.org/#explore?id={}".format(info["id"])
            return info["path"]
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default="./data",
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        default="imagenet",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)


    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    # model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        # print("Training network heads")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=40,
        #             layers='heads',
        #             augmentation=augmentation)

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=120,
        #             layers='4+',
        #             augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    # learning_rate=config.LEARNING_RATE / 10,
                    learning_rate=config.LEARNING_RATE ,
                    epochs=50,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset,  "val" )
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
