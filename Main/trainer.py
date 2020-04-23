import tensorflow as tf
import os
import numpy as np
import pandas as pd
from Helpers.dataset_handlers import read_tfr, save_tfr, get_feature_map
from Helpers.annotation_parsers import parse_voc_folder
from Helpers.anchors import k_means, generate_anchors
from Helpers.augmentor import DataAugment
from Config.augmentation_options import augmentations
from Main.models import V3Model
from Helpers.model_helpers import transform_images, transform_targets


class Trainer(V3Model):
    """
    Create a training instance.
    """
    def __init__(self, input_shape, classes_file, train_tf_record=None,
                 valid_tf_record=None, anchors=None, masks=None,
                 max_boxes=100, iou_threshold=0.5,
                 score_threshold=0.5):
        self.classes_file = classes_file
        self.class_names = [item.strip() for item in open(classes_file).readlines()]
        super().__init__(input_shape, len(self.class_names), anchors, masks, max_boxes,
                         iou_threshold, score_threshold)
        self.train_tf_record = train_tf_record
        self.valid_tf_record = valid_tf_record

    def generate_new_anchors(self, anchor_no, true_image_size):
        labels_frame = parse_voc_folder(os.path.join('..', 'Data', 'XML Labels'),
                                        os.path.join('..', 'Config', 'voc_conf.json'))
        relative_dims = np.array(list(zip(labels_frame['Relative Width'],
                                          labels_frame['Relative Height'])))
        centroids, _ = k_means(relative_dims, anchor_no, frame=labels_frame)
        self.anchors = generate_anchors(*true_image_size, centroids) / self.input_shape[0]

    def initialize_dataset(self, tf_record, batch_size):
        if not tf_record:
            return
        dataset = read_tfr(tf_record, self.classes_file, get_feature_map(), self.max_boxes)
        dataset = dataset.shuffle(buffer_size=512)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda x, y: (
            transform_images(x, self.input_shape[0]),
            transform_targets(y, self.anchors, self.masks, self.input_shape[0])))
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def augment_photos(new_dataset_conf, augmentation_batch_size=64,
                       new_augmentation_size=None):
        workers = new_dataset_conf.get('workers')
        converted_coordinates_file = new_dataset_conf.get('converted_coordinates_file')
        sequences = new_dataset_conf.get('sequences')
        labels_file = new_dataset_conf.get('labels_file')
        if not labels_file and not converted_coordinates_file:
            raise ValueError(f'No "labels_file" or "converted_coordinates_file"'
                             f' is found in new_dataset_conf')
        augment = DataAugment(labels_file, augmentations, workers or 32,
                              converted_coordinates_file)
        if not sequences:
            raise ValueError(f'No augmentation "sequences" were found in new_dataset_conf')
        augment.create_sequences(sequences)
        return augment.augment_photos_folder(
            augmentation_batch_size, new_augmentation_size)

    def train(self, epochs, batch_size, learning_rate, new_anchors_conf=None,
              new_dataset_conf=None, augmentation_batch_size=64, new_augmentation_size=None):
        if not new_dataset_conf and not self.train_tf_record:
            raise ValueError(f'No training dataset specified')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if new_anchors_conf:
            print(f'Generating new anchors ...')
            self.generate_new_anchors(new_anchors_conf['anchor_no'],
                                      new_anchors_conf['true_image_size'])
        self.create_models()
        if new_dataset_conf:
            print(f'Generating new dataset ...')
            dataset_name = new_dataset_conf.get('dataset_name')
            test_size = new_dataset_conf.get('test_size')
            labels_file = new_dataset_conf.get('labels_file')
            if not labels_file:
                raise ValueError(f'labels_file not provided')
            if not dataset_name:
                raise ValueError(f'"dataset_name" not found in new_dataset_conf')
            if new_dataset_conf.get('augmentation'):
                full_data = self.augment_photos(new_dataset_conf, augmentation_batch_size,
                                                new_augmentation_size)
                save_tfr(full_data, os.path.join('..', 'Data', 'TFRecords'), dataset_name, test_size,
                         self)
            if not new_dataset_conf.get('augmentation'):
                save_tfr(
                    pd.read_csv(labels_file), os.path.join(
                        '..', 'Data', 'TFRecords'), dataset_name, test_size, self)
        if not self.train_tf_record:
            raise ValueError(f'No training TFRecord specified')
        if not self.valid_tf_record:
            raise ValueError(f'No validation TFRecord specified')
        training_dataset = self.initialize_dataset(self.train_tf_record, batch_size)
        valid_dataset = self.initialize_dataset(self.valid_tf_record, batch_size)


if __name__ == '__main__':
    img_folder = '../../../beverly_hills/photos/'
    lbl_folder = '../../../beverly_hills/labels/'
    cls_file = '../Config/beverly_hills.txt'
    xx = Trainer((416, 416, 3), cls_file)
    dc = {'dataset_name': 'beverly_hills',
          'test_size': 0.2,
          'augmentation': True,
          'converted_coordinates_file': '/Users/emadboctor/Desktop/Code/'
                                        'yolov3-keras-tf2/Helpers/scratch/label_coordinates.csv',
          'sequences': [[{'sequence_group': 'meta', 'no': 5},
                           {'sequence_group': 'arithmetic', 'no': 3}],
                          [{'sequence_group': 'arithmetic', 'no': 2}]],
          'labels_file': '/Users/emadboctor/Desktop/beverly_hills/bh_labels.csv'
          }

    xx.train(50, 16, 1e-5, new_dataset_conf=dc)
