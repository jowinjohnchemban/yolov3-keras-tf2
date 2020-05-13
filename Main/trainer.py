import tensorflow as tf
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    TensorBoard,
    ModelCheckpoint,
)
from Helpers.dataset_handlers import read_tfr, save_tfr, get_feature_map
from Helpers.annotation_parsers import parse_voc_folder
from Helpers.anchors import k_means, generate_anchors
from Helpers.augmentor import DataAugment
from Config.augmentation_options import augmentations
from Main.models import V3Model
from Helpers.utils import transform_images, transform_targets
from Helpers.annotation_parsers import adjust_non_voc_csv
from Helpers.utils import calculate_loss, timer, default_logger
from Config.augmentation_options import preset_1
from Main.evaluator import Evaluator


class Trainer(V3Model):
    """
    Create a training instance.
    """

    def __init__(
        self,
        input_shape,
        classes_file,
        image_width,
        image_height,
        train_tf_record=None,
        valid_tf_record=None,
        anchors=None,
        masks=None,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
    ):
        """
        Initialize training.
        Args:
            input_shape: tuple, (n, n, c)
            classes_file: File containing class names \n delimited.
            image_width: Width of the original image.
            image_height: Height of the original image.
            train_tf_record: TFRecord file.
            valid_tf_record: TFRecord file.
            anchors: numpy array of (w, h) pairs.
            masks: numpy array of masks.
            max_boxes: Maximum boxes of the TFRecords provided(if any) or
                maximum boxes setting.
            iou_threshold: float, values less than the threshold are ignored.
            score_threshold: float, values less than the threshold are ignored.
        """
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
        self.image_folder = (
            Path(os.path.join('..', 'Data', 'Photos')).absolute().resolve()
        )
        self.image_width = image_width
        self.image_height = image_height

    def get_adjusted_labels(self, configuration):
        """
        Adjust labels according to given configuration.
        Args:
            configuration: A dictionary containing any of the following keys:
                - relative_labels
                - from_xml
                - adjusted_frame

        Returns:
            pandas DataFrame with adjusted labels.
        """
        labels_frame = None
        check = 0
        if configuration.get('relative_labels'):
            labels_frame = adjust_non_voc_csv(
                configuration['relative_labels'],
                self.image_folder,
                self.image_width,
                self.image_height,
            )
            check += 1
        if configuration.get('from_xml'):
            if check:
                raise ValueError(f'Got more than one configuration')
            labels_frame = parse_voc_folder(
                os.path.join('..', 'Data', 'XML Labels'),
                os.path.join('..', 'Config', 'voc_conf.json'),
            )
            labels_frame.to_csv(os.path.join('..', 'Output', 'parsed_from_xml.csv'), index=False)
            check += 1
        if configuration.get('adjusted_frame'):
            if check:
                raise ValueError(f'Got more than one configuration')
            labels_frame = pd.read_csv(configuration['adjusted_frame'])
            check += 1
        return labels_frame

    def generate_new_anchors(self, new_anchors_conf):
        """
        Create new anchors according to given configuration.
        Args:
            new_anchors_conf: A dictionary containing the following keys:
                - anchors_no
                and one of the following:
                    - relative_labels
                    - from_xml
                    - adjusted_frame

        Returns:
            None
        """
        anchor_no = new_anchors_conf.get('anchor_no')
        if not anchor_no:
            raise ValueError(f'No "anchor_no" found in new_anchors_conf')
        labels_frame = self.get_adjusted_labels(new_anchors_conf)
        relative_dims = np.array(
            list(
                zip(
                    labels_frame['Relative Width'],
                    labels_frame['Relative Height'],
                )
            )
        )
        centroids, _ = k_means(relative_dims, anchor_no, frame=labels_frame)
        self.anchors = (
            generate_anchors(self.image_width, self.image_height, centroids)
            / self.input_shape[0]
        )
        default_logger.info('Changed default anchors to generated ones')

    def generate_new_frame(self, new_dataset_conf):
        """
        Create new labels frame according to given configuration.
        Args:
            new_dataset_conf: A dictionary containing the following keys:
                - dataset_name
                and one of the following:
                    - relative_labels
                    - from_xml
                    - adjusted_frame
                    - coordinate_labels(optional in case of augmentation)
                - augmentation(optional)
                and this implies the following:
                    - sequences
                    - workers(optional, defaults to 32)
                    - batch_size(optional, defaults to 64)
                    - new_size(optional, defaults to None)

        Returns:
            pandas DataFrame adjusted for building the dataset containing
            labels or labels and augmented labels combined
        """
        if not new_dataset_conf.get('dataset_name'):
            raise ValueError('dataset_name not found in new_dataset_conf')
        labels_frame = self.get_adjusted_labels(new_dataset_conf)
        if new_dataset_conf.get('augmentation'):
            labels_frame = self.augment_photos(new_dataset_conf)
        return labels_frame

    def initialize_dataset(self, tf_record, batch_size, shuffle_buffer=512):
        """
        Initialize and prepare TFRecord dataset for training.
        Args:
            tf_record: TFRecord file.
            batch_size: int, training batch size
            shuffle_buffer: Buffer size for shuffling dataset.

        Returns:
            dataset.
        """
        dataset = read_tfr(
            tf_record, self.classes_file, get_feature_map(), self.max_boxes
        )
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda x, y: (
                transform_images(x, self.input_shape[0]),
                transform_targets(
                    y, self.anchors, self.masks, self.input_shape[0]
                ),
            )
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def augment_photos(new_dataset_conf):
        """
        Augment photos in self.image_paths
        Args:
            new_dataset_conf: A dictionary containing the following keys:
                one of the following:
                    - relative_labels
                    - from_xml
                    - adjusted_frame
                    - coordinate_labels(optional)
                and:
                    - sequences
                    - workers(optional, defaults to 32)
                    - batch_size(optional, defaults to 64)
                    - new_size(optional, defaults to None)

        Returns:
            pandas DataFrame with both original and augmented data.
        """
        sequences = new_dataset_conf.get('sequences')
        relative_labels = new_dataset_conf.get('relative_labels')
        coordinate_labels = new_dataset_conf.get('coordinate_labels')
        workers = new_dataset_conf.get('workers')
        batch_size = new_dataset_conf.get('batch_size')
        new_augmentation_size = new_dataset_conf.get('new_size')
        if not sequences:
            raise ValueError(f'"sequences" not found in new_dataset_conf')
        if not relative_labels:
            raise ValueError(f'No "relative_labels" found in new_dataset_conf')
        augment = DataAugment(
            relative_labels, augmentations, workers or 32, coordinate_labels
        )
        augment.create_sequences(sequences)
        return augment.augment_photos_folder(
            batch_size or 64, new_augmentation_size
        )

    @timer(default_logger)
    def evaluate(self, weights_file, merge, workers, shuffle_buffer,
                 min_overlaps, display_stats=True):
        default_logger.info('Starting evaluation ...')
        evaluator = Evaluator(
            self.input_shape, self.train_tf_record, self.valid_tf_record, self.classes_file,
            self.anchors, self.masks, self.masks, self.iou_threshold, self.score_threshold
        )
        predictions = evaluator.make_predictions(weights_file, merge, workers, shuffle_buffer)
        if isinstance(predictions, tuple):
            training_predictions, valid_predictions = predictions
            training_actual = pd.read_csv(os.path.join('..', 'Output', 'training_data.csv'))
            valid_actual = pd.read_csv(os.path.join('..', 'Output', 'test_data.csv'))
            training_stats, training_map = evaluator.calculate_map(
                training_predictions, training_actual, min_overlaps, display_stats)

            valid_stats, valid_map = evaluator.calculate_map(
                valid_predictions, valid_actual, min_overlaps, display_stats)
            return training_stats, training_map, valid_stats, valid_map
        actual_data = pd.read_csv(os.path.join('..', 'Output', 'full_data.csv'))
        stats, data_map = evaluator.calculate_map(
            predictions, actual_data, min_overlaps, display_stats)
        return stats, data_map

    @timer(default_logger)
    def train(
        self,
        epochs,
        batch_size,
        learning_rate,
        new_anchors_conf=None,
        new_dataset_conf=None,
        dataset_name=None,
        weights=None,
        evaluate=True,
        merge_evaluation=True,
        evaluation_workers=8,
        shuffle_buffer=512,
        min_overlaps=0.5,
        display_stats=True
    ):
        """
        Train on the dataset.
        Args:
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: non-negative value.
            new_anchors_conf: A dictionary containing anchor generation configuration.
            new_dataset_conf: A dictionary containing dataset generation configuration.
            dataset_name: Name of the dataset for model checkpoints.
            weights: .tf or .weights file
            evaluate: If False, the trained model will not be evaluated after training.
            merge_evaluation: If False, training and validation maps will
                be calculated separately.
            evaluation_workers: Parallel predictions.
            shuffle_buffer: Buffer size for shuffling datasets.
            min_overlaps: a float value between 0 and 1, or a dictionary
                containing each class in self.class_names mapped to its
                minimum overlap
            display_stats: If True and evaluate=True, evaluation statistics will be displayed.

        Returns:
            history object, pandas DataFrame with statistics, mAP score.
        """
        default_logger.info(f'Starting training ...')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            default_logger.info('GPU activated')
        if new_anchors_conf:
            print(f'Generating new anchors ...')
            self.generate_new_anchors(new_anchors_conf)
        self.create_models()
        if weights:
            self.load_weights(weights)
        if new_dataset_conf:
            print(f'Generating new dataset ...')
            test_size = new_dataset_conf.get('test_size')
            labels_frame = self.generate_new_frame(new_dataset_conf)
            save_tfr(
                labels_frame,
                os.path.join('..', 'Data', 'TFRecords'),
                new_dataset_conf['dataset_name'],
                test_size,
                self,
            )
        if not self.train_tf_record:
            raise ValueError(f'No training TFRecord specified')
        if not self.valid_tf_record:
            raise ValueError(f'No validation TFRecord specified')
        training_dataset = self.initialize_dataset(
            self.train_tf_record, batch_size, shuffle_buffer
        )
        valid_dataset = self.initialize_dataset(
            self.valid_tf_record, batch_size, shuffle_buffer
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = [
            calculate_loss(self.anchors[mask], self.classes, self.iou_threshold)
            for mask in self.masks
        ]
        self.training_model.compile(optimizer=optimizer, loss=loss)
        checkpoint_name = f'{dataset_name or "trained"}_model2.tf'
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            ModelCheckpoint(
                os.path.join('..', 'Models', checkpoint_name),
                verbose=1,
                save_weights_only=True,
            ),
            TensorBoard(log_dir=os.path.join('..', 'Logs')),
        ]
        history = self.training_model.fit(
            training_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=valid_dataset,
        )
        default_logger.info('Training complete')
        if evaluate:
            evaluations = self.evaluate(os.path.join('..', 'Models', checkpoint_name),
                                        merge_evaluation, evaluation_workers,
                                        shuffle_buffer, min_overlaps, display_stats)
            return evaluations, history
        return history


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
    tr = Trainer(
        (416, 416, 3),
        '../Config/beverly_hills.txt',
        1344,
        756,
        anchors=anc,
        train_tf_record='/content/drive/My Drive/beverly_hills_train.tfrecord',
        valid_tf_record='/content/drive/My Drive/beverly_hills_test.tfrecord',
    )
    dt = {
        'relative_labels': '../Data/bh_labels.csv',
        'dataset_name': 'beverly_hills',
        'test_size': 0.2,
        'sequences': preset_1,
        'augmentation': True,
    }
    tr.train(150, 8, 1e-3, dataset_name='beverly_hills')
