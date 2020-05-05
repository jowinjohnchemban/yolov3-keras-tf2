import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from Main.models import V3Model
from Helpers.dataset_handlers import read_tfr, get_feature_map
from Helpers.utils import transform_images, get_detection_data
from Helpers.annotation_parsers import adjust_non_voc_csv


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

    @staticmethod
    def get_iou(box_true, box_predicted):
        x1, y1, x2, y2 = box_true
        x1p, y1p, x2p, y2p = box_predicted
        if not all([x2 > x1, y2 > y1, x2p > x1p, y2p > y1p]):
            return 0
        far_x = np.min([x2, x2p])
        near_x = np.max([x1, x1p])
        far_y = np.min([y2, y2p])
        near_y = np.max([y1, y1p])
        inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
        true_box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        pred_box_area = (x2p - x1p + 1) * (y2p - y1p + 1)
        iou = inter_area / (true_box_area + pred_box_area - inter_area)
        return iou

    def calculate_overlaps(self, detections, actual):
        calculations = []
        detection_groups = detections.groupby('image')
        actual_groups = actual.groupby('Image Path')
        for item1, item2 in zip(actual_groups, detection_groups):
            for detected_index, detected_row in item2[1].iterrows():
                detected_coordinates = detected_row.values[2: 6]
                detected_overlaps = []
                coords = []
                for actual_index, actual_row in item1[1].iterrows():
                    actual_coordinates = actual_row.values[4: 8]
                    detected_overlaps.append((
                        self.get_iou(actual_coordinates, detected_coordinates)))
                    coords.append(actual_coordinates)
                detected_row['max_iou'] = max(detected_overlaps)
                x1, y1, x2, y2 = coords[int(np.argmax(detected_overlaps))]
                for match, value in zip([f'{item}_match'
                                         for item in ['x1', 'y1', 'x2', 'y2']],
                                        [x1, y1, x2, y2]):
                    detected_row[match] = value
                calculations.append(detected_row)
        return pd.DataFrame(calculations)

    def get_counts(self, prediction_file, actual_file):
        detection_data = pd.read_csv(prediction_file)
        width, height = detection_data.iloc[0][['image_width', 'image_height']]
        actual_data = adjust_non_voc_csv(actual_file, '', width, height)
        for object_name in self.class_names:
            detections = detection_data[detection_data['object_name'] == object_name]
            actual = actual_data[actual_data['Object Name'] == object_name]
            calculated = self.calculate_overlaps(detections, actual)
            print(calculated)


if __name__ == '__main__':
    anc = np.array([[58, 90], [695, 274], [262, 196], [62, 132], [152, 118],
                    [185, 349], [50, 105], [531, 455], [248, 427]])
    ev = Evaluator((416, 416, 3), '../../bhills_train.tfrecord',
                   '../../bhills_test.tfrecord', '../Config/beverly_hills.txt', anc)
    # ev.make_predictions('../../../beverly_hills/models/beverly_hills_model.tf', merge=True)
    ev.get_counts('../Caches/full_dataset_predictions.csv', '../Data/bh_labels.csv')

