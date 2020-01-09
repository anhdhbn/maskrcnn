import skimage.draw
import os 

def mkdir(path):
    try:
        os.mkdir(dirname)
    except:
        pass

def detect(model, ROOT_DIR):
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset = CocoDataset()
    dataset.load_coco("./data", "val")
    dataset.prepare()
    from mrcnn import visualize

    folder = ROOT_DIR + "samsung/out"
    mkdir(folder)

    for image_id in dataset.image_ids:
        # Load image and run detection
        info  = dataset.image_info[image_id]

        image = skimage.io.imread(info["path"])
        print("Running on {}".format(info["path"]))
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(folder, dataset.image_info[image_id]["id"]))
