import tensorflow as tf
from tensorflow.keras.losses import (
    sparse_categorical_crossentropy,
    binary_crossentropy,
)
import logging
from logging import handlers
from time import perf_counter
import os
import numpy as np
import pandas as pd
from xml.etree.ElementTree import SubElement
from xml.etree import ElementTree
from lxml import etree


def get_logger():
    formatter = logging.Formatter(
        '%(asctime)s %(name)s.%(funcName)s +%(lineno)s: '
        '%(levelname)-8s [%(process)d] %(message)s'
    )
    logger = logging.getLogger('session_log')
    logger.setLevel(logging.DEBUG)
    file_title = os.path.join('Logs', 'session.log')
    if 'Logs' not in os.listdir():
        file_title = f'{os.path.join("..", file_title)}'
    file_handler = handlers.RotatingFileHandler(file_title, backupCount=10)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


default_logger = get_logger()


def timer(logger):
    def timed(func):
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            total_time = perf_counter() - start_time
            if logger is not None:
                logger.info(
                    f'{func.__name__} execution time: ' f'{total_time} seconds'
                )
            if result is not None:
                return result

        return wrapper

    return timed


def ratios_to_coordinates(bx, by, bw, bh, width, height):
    """
    Convert relative coordinates to actual coordinates.
    Args:
        bx: Relative center x coordinate.
        by: Relative center y coordinate.
        bw: Relative box width.
        bh: Relative box height.
        width: Image batch width.
        height: Image batch height.

    Return:
        x1: x coordinate.
        y1: y coordinate.
        x2: x1 + Bounding box width.
        y2: y1 + Bounding box height.
    """
    w, h = bw * width, bh * height
    x, y = bx * width + (w / 2), by * height + (h / 2)
    return x, y, x + w, y + h


def transform_images(x_train, size):
    """
    Resize image tensor.
    Args:
        x_train: Image tensor.
        size: new (width, height)
    """
    x_train = tf.image.resize(x_train, (size, size))
    return x_train / 255


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    n = tf.shape(y_true)[0]
    y_true_out = tf.zeros(
        (n, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6)
    )
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(n):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32)
            )
            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]
                )
                idx += 1
    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack()
    )


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(
        tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1)
    )
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1]
    )
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
    y_train = tf.concat([y_train, anchor_idx], axis=-1)
    for anchor_idxs in anchor_masks:
        y_outs.append(
            transform_targets_for_output(y_train, grid_size, anchor_idxs)
        )
        grid_size *= 2
    return tuple(y_outs)


def broadcast_iou(box_1, box_2):
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    int_w = tf.maximum(
        tf.minimum(box_1[..., 2], box_2[..., 2])
        - tf.maximum(box_1[..., 0], box_2[..., 0]),
        0,
    )
    int_h = tf.maximum(
        tf.minimum(box_1[..., 3], box_2[..., 3])
        - tf.maximum(box_1[..., 1], box_2[..., 1]),
        0,
    )
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (
        box_1[..., 3] - box_1[..., 1]
    )
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (
        box_2[..., 3] - box_2[..., 1]
    )
    return int_area / (box_1_area + box_2_area - int_area)


def get_boxes(pred, anchors, classes):
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, object_probability, class_probabilities = tf.split(
        pred, (2, 2, 1, classes), axis=-1
    )
    box_xy = tf.sigmoid(box_xy)
    object_probability = tf.sigmoid(object_probability)
    class_probabilities = tf.sigmoid(class_probabilities)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(
        grid_size, tf.float32
    )
    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, object_probability, class_probabilities, pred_box


def calculate_loss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        pred_box, pred_obj, pred_class, pred_xywh = get_boxes(
            y_pred, anchors, classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1
        )
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(
            grid, tf.float32
        )
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(
            tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh
        )
        obj_mask = tf.squeeze(true_obj, -1)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                broadcast_iou(
                    x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))
                ),
                axis=-1,
            ),
            (pred_box, true_box, obj_mask),
            tf.float32,
        )
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        xy_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        )
        wh_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        )
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class
        )
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


def add_xml_path(xml_file, path):
    tree = ElementTree.parse(xml_file)
    top = tree.getroot()
    folder_tag = tree.find('folder')
    folder_tag.text = path
    file_name_tag = tree.find('filename')
    path_tag = SubElement(top, 'path')
    path_tag.text = os.path.join(folder_tag.text, file_name_tag.text)
    rough_string = ElementTree.tostring(top, 'utf8')
    root = etree.fromstring(rough_string)
    pretty = etree.tostring(root, pretty_print=True, encoding='utf-8').replace(
        '  '.encode(), '\t'.encode()
    )
    os.remove(xml_file)
    with open(xml_file, 'wb') as output:
        output.write(pretty)


def get_detection_data(image, image_name, outputs, class_names):
    nums = outputs[-1]
    boxes, scores, classes = [
        item[0][: int(nums)].numpy() for item in outputs[:-1]
    ]
    w, h = np.flip(image.shape[0: 2])
    data = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    data[['x1', 'x2']] = (data[['x1', 'x2']] * w).astype('int64')
    data[['y1', 'y2']] = (data[['y1', 'y2']] * h).astype('int64')
    data['object_name'] = np.array(class_names)[classes.astype('int64')]
    data['image'] = image_name
    data['score'] = scores
    data['image_width'] = w
    data['image_height'] = h
    data = data[
        [
            'image',
            'object_name',
            'x1',
            'y1',
            'x2',
            'y2',
            'score',
            'image_width',
            'image_height',
        ]
    ]
    return data


def activate_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        default_logger.info('GPU activated')
