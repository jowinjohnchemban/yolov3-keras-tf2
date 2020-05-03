import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from Main.models import V3Model
from Helpers.dataset_handlers import read_tfr, get_feature_map
from Helpers.utils import transform_images, get_detection_data


class Evaluator(V3Model):
    def __init__(self, input_shape, train_tf_record, valid_tf_record, classes_file,
                 anchors=None, masks=None, max_boxes=100, iou_threshold=0.5,
                 score_threshold=0.5):
        self.classes_file = classes_file
        self.class_names = [item.strip() for item in open(classes_file).readlines()]
        super().__init__(input_shape, len(self.class_names), anchors, masks, max_boxes,
                         iou_threshold, score_threshold)
        self.train_tf_record = train_tf_record
        self.valid_tf_record = valid_tf_record
        train_dataset_size = sum(1 for _ in tf.data.TFRecordDataset(train_tf_record))
        valid_dataset_size = sum(1 for _ in tf.data.TFRecordDataset(valid_tf_record))
        self.dataset_size = train_dataset_size + valid_dataset_size
        self.predicted = 0

    def predict_image(self, image_data, features):
        image_path = bytes.decode(features['image_path'].numpy())
        image_name = os.path.basename(image_path)
        image = tf.expand_dims(image_data, 0)
        resized = transform_images(image, 416)
        outs = self.inference_model(resized)
        adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
        return get_detection_data(adjusted, image_name, outs, self.class_names), image_name

    def predict_dataset(self, dataset, workers=16):
        predictions = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_predictions = {
                executor.submit(self.predict_image, img_data, features):
                    features['image_path'] for img_data, labels, features in iter(dataset)}
            for future_prediction in as_completed(future_predictions):
                result, completed_image = future_prediction.result()
                predictions.append(result)
                completed = f'{self.predicted}/{self.dataset_size}'
                percent = (self.predicted / self.dataset_size) * 100
                print(f'\rpredicting {completed_image} {completed}\t{percent}% completed', end='')
                self.predicted += 1
        return pd.concat(predictions)

    def make_predictions(self, trained_weights, merge=False, workers=16):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.create_models()
        self.load_weights(trained_weights)
        features = get_feature_map()
        train_dataset = read_tfr(self.train_tf_record, self.classes_file, features,
                                 self.max_boxes, get_features=True)
        valid_dataset = read_tfr(self.valid_tf_record, self.classes_file, features,
                                 self.max_boxes, get_features=True)
        train_dataset.shuffle(512)
        valid_dataset.shuffle(512)
        train_predictions = self.predict_dataset(train_dataset, workers)
        valid_predictions = self.predict_dataset(valid_dataset, workers)
        if merge:
            predictions = pd.concat([train_predictions, valid_predictions])
            save_path = os.path.join('..', 'Caches', 'full_dataset_predictions.csv')
            predictions.to_csv(save_path, index=False)
            return predictions
        train_path = os.path.join('..', 'Caches', 'train_dataset_predictions.csv')
        valid_path = os.path.join('..', 'Caches', 'valid_dataset_predictions.csv')
        train_predictions.to_csv(train_path, index=False)
        valid_predictions.to_csv(valid_path, index=False)
        return train_predictions, valid_predictions


# anc = np.array([[58, 90], [695, 274], [262, 196], [62, 132], [152, 118],
#                     [185, 349], [50, 105], [531, 455], [248, 427]])
# mod = V3Model((416, 416, 3), 16, anc, max_boxes=100)
# tr, inf = mod.create_models()
# mod.load_weights('../../../beverly_hills/models/beverly_hills_model.tf')
# class_names = [c.strip().replace(" ", "_").lower()
#                for c in open('../Config/beverly_hills.txt').readlines()]
# train_dataset = read_tfr('../../bhills_train.tfrecord', '../Config/beverly_hills.txt',
#                          get_feature_map(), 100, get_features=True)
# train_dataset = train_dataset.shuffle(512)
# detections = []
#
# t1 = perf_counter()

# for image_data, labels, features in iter(train_dataset):
#     image_path = bytes.decode(features['image_path'].numpy())
#     image_name = os.path.basename(image_path)
#     print(f'Predicting {image_name}')
#     image = tf.expand_dims(image_data, 0)
#     resized = transform_images(image, 416)
#     outs = inf(resized)
#     adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
#     current_detections = get_detection_data(adjusted, image_name, outs, class_names)
#     detections.append(current_detections)
# fr = pd.concat(detections)
# fr.to_csv('../../../preds.csv', index=False)
# t2 = perf_counter()
# print(f'{t2 - t1} seconds')
#
#
#
#
#
# # img_raw, label = next(iter(train_dataset.take(1)))
# # xyz = label.numpy()
# # cv2.imwrite('../../../test_image.png', img_raw.numpy())
#
#
# def calculate_ratios(x1, y1, x2, y2, width, height):
#     """
#     Calculate relative object ratios in the labeled image.
#     Args:
#         x1: Start x coordinate.
#         y1: Start y coordinate.
#         x2: End x coordinate.
#         y2: End y coordinate.
#         width: Bounding box width.
#         height: Bounding box height.
#
#     Return:
#         bx: Relative center x coordinate.
#         by: Relative center y coordinate.
#         bw: Relative box width.
#         bh: Relative box height.
#     """
#     box_width = abs(x2 - x1)
#     box_height = abs(y2 - y1)
#     bx = 1 - ((width - min(x1, x2) + (box_width / 2)) / width)
#     by = 1 - ((height - min(y1, y2) + (box_height / 2)) / height)
#     bw = box_width / width
#     bh = box_height / height
#     return bx, by, bw, bh


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


if __name__ == '__main__':
    anc = np.array([[58, 90], [695, 274], [262, 196], [62, 132], [152, 118],
                    [185, 349], [50, 105], [531, 455], [248, 427]])
    ev = Evaluator((416, 416, 3), '../../bhills_train.tfrecord',
                   '../../bhills_test.tfrecord', '../Config/beverly_hills.txt', anc)
    ev.make_predictions('../../../beverly_hills/models/beverly_hills_model.tf', merge=True)

