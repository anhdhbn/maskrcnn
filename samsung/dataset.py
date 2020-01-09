############################################################
#  Dataset
############################################################
from mrcnn import model as modellib, utils

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

        self.add_class("coco", 1, "loai 1")
        self.add_class("coco", 2, "loai 2")
        image_dir = "{}/{}".format(dataset_dir, subset)


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
                "class": int(data[-1])+1
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

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        # instance_masks = []
        class_ids = []

        for i, p in enumerate(info["polygons"]):
            class_id = p["class"]
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            class_ids.append(class_id)
        return mask.astype(np.bool), np.asarray(class_ids,  dtype=np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info["path"]
        else:
            super(CocoDataset, self).image_reference(image_id)
