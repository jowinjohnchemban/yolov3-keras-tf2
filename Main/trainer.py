import tensorflow as tf
import os
import numpy as np
from Helpers.dataset_handlers import read_tfr, save_tfr, get_feature_map
from Helpers.annotation_parsers import parse_voc_folder
from Helpers.anchors import k_means, generate_anchors
from Helpers.augmentor import DataAugment
from Main.models import V3Model


class Trainer(V3Model):
    """
    Create a training instance.
    """
    def __init__(self, input_shape, classes_file, train_data_set=None,
                 valid_data_set=None, anchors=None, masks=None,
                 max_boxes=100, iou_threshold=0.5,
                 score_threshold=0.5):
        self.class_names = [item.strip() for item in open(classes_file).readlines()]
        super().__init__(input_shape, len(self.class_names), anchors, masks, max_boxes,
                         iou_threshold, score_threshold)
        self.train_data_set = train_data_set
        self.valid_data_set = valid_data_set

    def generate_new_anchors(self, anchor_no, true_image_size):
        labels_frame = parse_voc_folder('../Data/XML Labels', '../Config/voc_conf.json')
        relative_dims = np.array(list(zip(labels_frame['Relative Width'],
                                          labels_frame['Relative Height'])))
        centroids, _ = k_means(relative_dims, anchor_no, frame=labels_frame)
        self.anchors = generate_anchors(*true_image_size, centroids) / self.input_shape[0]

    def train(self, epochs, batch_size, learning_rate, new_anchors_conf=None,
              new_dataset_conf=None):
        if not os.listdir(os.path.join('..', 'Data', 'Photos')) and (
                not os.listdir(os.path.join('..', 'Data', 'TFRecords'))):
            raise ValueError('No photos or TFRecords were found in Data folder')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if new_anchors_conf:
            print(f'Generating new anchors ...')
            self.generate_new_anchors(new_anchors_conf['anchor_no'],
                                      new_anchors_conf['true_image_size'])
        self.create_models()
        if not new_dataset_conf and not os.listdir('../Data/TFRecords'):
            raise ValueError('No TFRecords were found in Data/TFRecords '
                             'and new_dataset_conf is None')
        if new_dataset_conf:
            pass


if __name__ == '__main__':
    img_folder = '../../../beverly_hills/photos/'
    lbl_folder = '../../../beverly_hills/labels/'
    cls_file = '../Config/beverly_hills.txt'
    x = Trainer((416, 416, 3), cls_file, img_folder, lbl_folder)
    x.train(50, 16, 1e-5, {'anchor_no': 9, 'true_image_size': (1344, 756)})
