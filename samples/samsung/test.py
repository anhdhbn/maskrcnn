import numpy as np
from PIL import Image, ImageDraw

all_points_y = [191, 191, 192, 193, 193, 193, 192, 192, 194, 194, 192]
all_points_x = [424, 432, 442, 452, 461, 463, 463, 470, 465, 443, 424]
# img = Image.new('L', (624, 744), 0)
# polygon = list(zip(all_points_x, all_points_y))
# print(polygon)
# ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
# mask = np.array(img)
# print(mask)
# _idx = np.sum(np.stack([mask], axis=2).astype(np.bool), axis=(0, 1)) > 0
# print(_idx)

import skimage.draw
class_ids = [0]
mask = np.zeros([744, 624, 1],
                        dtype=np.uint8)
rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
mask[rr, cc, 0] = 1

mask.astype(np.bool)
class_ids = np.asarray(class_ids, dtype=np.int32)
_idx = np.sum(mask, axis=(0, 1)) > 0
mask = mask[:, :, _idx]

class_ids = class_ids[_idx]
print(_idx, class_ids)