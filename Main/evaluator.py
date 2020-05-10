import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from Main.models import V3Model
from Helpers.dataset_handlers import read_tfr, get_feature_map
from Helpers.utils import transform_images, get_detection_data, default_logger, timer
from Helpers.annotation_parsers import adjust_non_voc_csv


class Evaluator(V3Model):
    def __init__(
        self,
        input_shape,
        train_tf_record,
        valid_tf_record,
        classes_file,
        anchors=None,
        masks=None,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
    ):
        self.classes_file = classes_file
        self.class_names = [
            item.strip() for item in open(classes_file).readlines()
        ]
        super().__init__(
            input_shape,
            len(self.class_names),
            anchors,
            masks,
            max_boxes,
            iou_threshold,
            score_threshold,
        )
        self.train_tf_record = train_tf_record
        self.valid_tf_record = valid_tf_record
        train_dataset_size = sum(
            1 for _ in tf.data.TFRecordDataset(train_tf_record)
        )
        valid_dataset_size = sum(
            1 for _ in tf.data.TFRecordDataset(valid_tf_record)
        )
        self.dataset_size = train_dataset_size + valid_dataset_size
        self.predicted = 0

    def predict_image(self, image_data, features):
        image_path = bytes.decode(features['image_path'].numpy())
        image_name = os.path.basename(image_path)
        image = tf.expand_dims(image_data, 0)
        resized = transform_images(image, 416)
        outs = self.inference_model(resized)
        adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
        return (
            get_detection_data(adjusted, image_name, outs, self.class_names),
            image_name,
        )

    def predict_dataset(self, dataset, workers=16):
        predictions = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_predictions = {
                executor.submit(
                    self.predict_image, img_data, features
                ): features['image_path']
                for img_data, labels, features in iter(dataset)
            }
            for future_prediction in as_completed(future_predictions):
                result, completed_image = future_prediction.result()
                predictions.append(result)
                completed = f'{self.predicted}/{self.dataset_size}'
                percent = (self.predicted / self.dataset_size) * 100
                print(
                    f'\rpredicting {completed_image} {completed}\t{percent}% completed',
                    end='',
                )
                self.predicted += 1
        return pd.concat(predictions)

    @timer(default_logger)
    def make_predictions(self, trained_weights, merge=False, workers=16):
        self.create_models()
        self.load_weights(trained_weights)
        features = get_feature_map()
        train_dataset = read_tfr(
            self.train_tf_record,
            self.classes_file,
            features,
            self.max_boxes,
            get_features=True,
        )
        valid_dataset = read_tfr(
            self.valid_tf_record,
            self.classes_file,
            features,
            self.max_boxes,
            get_features=True,
        )
        train_dataset.shuffle(512)
        valid_dataset.shuffle(512)
        train_predictions = self.predict_dataset(train_dataset, workers)
        valid_predictions = self.predict_dataset(valid_dataset, workers)
        if merge:
            predictions = pd.concat([train_predictions, valid_predictions])
            save_path = os.path.join(
                '..', 'Caches', 'full_dataset_predictions.csv'
            )
            predictions.to_csv(save_path, index=False)
            return predictions
        train_path = os.path.join(
            '..', 'Caches', 'train_dataset_predictions.csv'
        )
        valid_path = os.path.join(
            '..', 'Caches', 'valid_dataset_predictions.csv'
        )
        train_predictions.to_csv(train_path, index=False)
        valid_predictions.to_csv(valid_path, index=False)
        return train_predictions, valid_predictions

    @staticmethod
    def get_area(frame, columns):
        x1, y1, x2, y2 = [frame[column] for column in columns]
        return (x2 - x1) * (y2 - y1)

    def get_true_positives(self, detections, actual, min_overlaps):
        actual = actual.rename(
            columns={'Image Path': 'image', 'Object Name': 'object_name'}
        )
        random_gen = np.random.default_rng()
        if 'detection_key' not in detections.columns:
            detection_keys = random_gen.choice(
                len(detections), size=len(detections), replace=False
            )
            detections['detection_key'] = detection_keys
        total_frame = actual.merge(detections, on=['image', 'object_name'])
        total_frame['x_max_common'] = total_frame[['X_max', 'x2']].min(1)
        total_frame['x_min_common'] = total_frame[['X_min', 'x1']].max(1)
        total_frame['y_max_common'] = total_frame[['Y_max', 'y2']].min(1)
        total_frame['y_min_common'] = total_frame[['Y_min', 'y1']].max(1)
        true_intersect = (
            total_frame['x_max_common'] > total_frame['x_min_common']
        ) & (total_frame['y_max_common'] > total_frame['y_min_common'])
        total_frame = total_frame[true_intersect]
        actual_areas = self.get_area(
            total_frame, ['X_min', 'Y_min', 'X_max', 'Y_max']
        )
        predicted_areas = self.get_area(total_frame, ['x1', 'y1', 'x2', 'y2'])
        intersect_areas = self.get_area(
            total_frame,
            ['x_min_common', 'y_min_common', 'x_max_common', 'y_max_common'],
        )
        iou_areas = intersect_areas / (
            actual_areas + predicted_areas - intersect_areas
        )
        total_frame['iou'] = iou_areas
        if isinstance(min_overlaps, float):
            return total_frame[total_frame['iou'] >= min_overlaps]
        if isinstance(min_overlaps, dict):
            class_data = [
                (name, total_frame[total_frame['object_name'] == name])
                for name in self.class_names
            ]
            thresholds = [min_overlaps[item[0]] for item in class_data]
            frames = [
                item[1][item[1]['iou'] >= threshold]
                for (item, threshold) in zip(class_data, thresholds)
                if not item[1].empty
            ]
            return pd.concat(frames)

    @staticmethod
    def get_false_positives(detections, true_positive):
        keys_before = detections['detection_key'].values
        keys_after = true_positive['detection_key'].values
        false_keys = np.where(np.isin(keys_before, keys_after, invert=True))
        false_keys = keys_before[false_keys]
        false_positives = detections.set_index('detection_key').loc[false_keys]
        return false_positives.reset_index()

    @staticmethod
    def combine_results(true_positive, false_positive):
        true_positive['true_positive'] = 1
        true_positive['false_positive'] = 0
        true_positive = true_positive[
            [
                'image',
                'object_name',
                'score',
                'x_min_common',
                'y_min_common',
                'x_max_common',
                'y_max_common',
                'iou',
                'image_width',
                'image_height',
                'true_positive',
                'false_positive',
                'detection_key',
            ]
        ]
        true_positive = true_positive.rename(
            columns={
                'x_min_common': 'x1',
                'y_min_common': 'y1',
                'x_max_common': 'x2',
                'y_max_common': 'y2',
            }
        )
        false_positive['iou'] = 0
        false_positive['true_positive'] = 0
        false_positive['false_positive'] = 1
        false_positive = false_positive[
            [
                'image',
                'object_name',
                'score',
                'x1',
                'y1',
                'x2',
                'y2',
                'iou',
                'image_width',
                'image_height',
                'true_positive',
                'false_positive',
                'detection_key',
            ]
        ]
        return pd.concat([true_positive, false_positive])

    def calculate_stats(self, actual_data, detection_data, true_positives, false_positives, combined):
        class_stats = []
        for class_name in self.class_names:
            stats = dict()
            stats['Class Name'] = class_name
            stats['Average Precision'] = (combined[combined['object_name'] == class_name]
                                          ['average_precision'].sum() * 100)
            stats['Actual Detections'] = len(actual_data[actual_data["Object Name"] == class_name])
            stats['Detections'] = len(detection_data[detection_data["object_name"] == class_name])
            stats['True Positives'] = len(true_positives[true_positives["object_name"] == class_name])
            stats['False Positives'] = len(false_positives[false_positives["object_name"] == class_name])
            stats['Combined'] = len(combined[combined["object_name"] == class_name])
            class_stats.append(stats)
        return pd.DataFrame(class_stats)

    @staticmethod
    def calculate_ap(combined, total_actual):
        combined = combined.sort_values(by='score', ascending=False).reset_index(drop=True)
        combined['acc_tp'] = combined['true_positive'].cumsum()
        combined['acc_fp'] = combined['false_positive'].cumsum()
        combined['precision'] = combined['acc_tp'] / (combined['acc_tp'] + combined['acc_fp'])
        combined['recall'] = combined['acc_tp'] / total_actual
        combined['m_pre1'] = combined['precision'].shift(1, fill_value=0)
        combined['m_pre'] = combined[['m_pre1', 'precision']].max(axis=1)
        combined['m_rec1'] = combined['recall'].shift(1, fill_value=0)
        combined.loc[combined['m_rec1'] != combined['recall'], 'valid_m_rec'] = 1
        combined['average_precision'] = (combined['recall'] - combined['m_rec1']) * combined['m_pre']
        return combined

    @timer(default_logger)
    def get_all_stats(
        self, prediction_file, actual_file, min_overlaps, display_stats=False
    ):
        detection_data = pd.read_csv(prediction_file)
        width, height = detection_data.iloc[0][['image_width', 'image_height']]
        actual_data = adjust_non_voc_csv(actual_file, '', width, height)
        class_counts = actual_data['Object Name'].value_counts().to_dict()
        true_positives = self.get_true_positives(
            detection_data, actual_data, min_overlaps
        )
        false_positives = self.get_false_positives(
            detection_data, true_positives
        )
        combined = self.combine_results(true_positives, false_positives)
        class_groups = combined.groupby('object_name')
        calculated = pd.concat([self.calculate_ap(group, class_counts.get(object_name))
                                for object_name, group in class_groups])
        stats = self.calculate_stats(
                actual_data, detection_data, true_positives, false_positives, calculated)
        if display_stats:
            pd.set_option('display.max_rows', None,
                          'display.max_columns', None,
                          'display.width', None)
            print(stats)
            pd.reset_option('display.[max_rows, max_columns, width]')
        return stats


if __name__ == '__main__':
    anc = np.array(
        [
            [58, 90],
            [695, 274],
            [262, 196],
            [62, 132],
            [152, 118],
            [185, 349],
            [50, 105],
            [531, 455],
            [248, 427],
        ]
    )
    ev = Evaluator(
        (416, 416, 3),
        '../../bhills_train.tfrecord',
        '../../bhills_test.tfrecord',
        '../Config/beverly_hills.txt',
        anc,
    )
    # ev.make_predictions('../../../beverly_hills/models/beverly_hills_model.tf', merge=True)
    ovs = {
        'Car': 0.55,
        'Street Sign': 0.5,
        'Palm Tree': 0.5,
        'Street Lamp': 0.5,
        'Minivan': 0.5,
        'Traffic Lights': 0.5,
        'Pedestrian': 0.5,
        'Fire Hydrant': 0.5,
        'Flag': 0.5,
        'Trash Can': 0.5,
        'Bicycle': 0.5,
        'Bus': 0.5,
        'Pickup Truck': 0.5,
        'Road Block': 0.5,
        'Delivery Truck': 0.5,
        'Motorcycle': 0.5,
    }
    ev.get_all_stats(
        '../Caches/full_dataset_predictions.csv',
        '../Data/bh_labels.csv',
        min_overlaps=ovs,
        display_stats=True,
    )
