import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from Main.models import V3Model
from Helpers.dataset_handlers import read_tfr, get_feature_map
from Helpers.utils import transform_images, get_detection_data


anc = np.array([[58, 90], [695, 274], [262, 196], [62, 132], [152, 118],
                    [185, 349], [50, 105], [531, 455], [248, 427]])
mod = V3Model((416, 416, 3), 16, anc, max_boxes=100)
tr, inf = mod.create_models()
mod.load_weights('../../../beverly_hills/models/beverly_hills_model.tf')
class_names = [c.strip().replace(" ", "_").lower()
               for c in open('../Config/beverly_hills.txt').readlines()]
train_dataset = read_tfr('../../bhills_train.tfrecord', '../Config/beverly_hills.txt',
                         get_feature_map(), 100, get_features=True)
train_dataset = train_dataset.shuffle(512)
detections = []
actual = []
for image_data, labels, features in iter(train_dataset):
    actual_frame = pd.DataFrame()
    image_path = bytes.decode(features['image_path'].numpy())
    image_name = os.path.basename(image_path)
    image_width = int(features['image_width'])
    image_height = int(features['image_height'])
    object_names = tf.sparse.to_dense(features['object_name']).numpy().astype('U13')
    x_min = (tf.sparse.to_dense(features['x_min']).numpy() * image_width).astype('int64')
    y_min = (tf.sparse.to_dense(features['y_min']).numpy() * image_height).astype('int64')
    x_max = (tf.sparse.to_dense(features['x_max']).numpy() * image_width).astype('int64')
    y_max = (tf.sparse.to_dense(features['y_max']).numpy() * image_height).astype('int64')
    object_id = tf.sparse.to_dense(features['object_id']).numpy().astype('int64')
    image = tf.expand_dims(image_data, 0)
    resized = transform_images(image, 416)
    adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
    current_detections = get_detection_data(adjusted, inf(adjusted), class_names)
    detections.extend(current_detections)
    actual_frame['Object Name'] = object_names
    actual_frame['Image Width'] = image_width
    actual_frame['Image Height'] = image_height
    actual_frame['X_min'] = x_min
    actual_frame['Y_min'] = y_min
    actual_frame['X_max'] = x_max
    actual_frame['Y_max'] = y_max
    actual_frame['Object ID'] = object_id
    actual.append(actual_frame)
    print(actual)
    break


# img_raw, label = next(iter(train_dataset.take(1)))
# xyz = label.numpy()
# cv2.imwrite('../../../test_image.png', img_raw.numpy())


def calculate_ratios(x1, y1, x2, y2, width, height):
    """
    Calculate relative object ratios in the labeled image.
    Args:
        x1: Start x coordinate.
        y1: Start y coordinate.
        x2: End x coordinate.
        y2: End y coordinate.
        width: Bounding box width.
        height: Bounding box height.

    Return:
        bx: Relative center x coordinate.
        by: Relative center y coordinate.
        bw: Relative box width.
        bh: Relative box height.
    """
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    bx = 1 - ((width - min(x1, x2) + (box_width / 2)) / width)
    by = 1 - ((height - min(y1, y2) + (box_height / 2)) / height)
    bw = box_width / width
    bh = box_height / height
    return bx, by, bw, bh


# zyx = []
# cls = [item.strip() for item in open('../Config/beverly_hills.txt').readlines()]
# for item in xyz:
#     if not item.any():
#         break
#     x1, y1, x2, y2, obj = item
#     x1 *= 416
#     y1 *= 416
#     x2 *= 416
#     y2 *= 416
#     bx, by, bw, bh = calculate_ratios(x1, y1, x2, y2, 416, 416)
#     row = ['test_image.png', cls[int(obj)], obj, bx, by, bw, bh]
#     zyx.append(row)
# fr = pd.DataFrame(zyx, columns=['Image', 'Object Name', 'Object Index', 'bx', 'by', 'bw', 'bh'])
# fr.to_csv('../../../test_frame.csv', index=False)

