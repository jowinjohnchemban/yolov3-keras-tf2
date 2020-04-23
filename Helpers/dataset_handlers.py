import tensorflow as tf
import pandas as pd
import numpy as np
import hashlib
import os


def get_feature_map():
    features = {
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'image_file': tf.io.FixedLenFeature([], tf.string),
        'image_key': tf.io.FixedLenFeature([], tf.string),
        'image_data': tf.io.FixedLenFeature([], tf.string),
        'image_format': tf.io.FixedLenFeature([], tf.string),
        'x_min': tf.io.VarLenFeature(tf.float32),
        'y_min': tf.io.VarLenFeature(tf.float32),
        'x_max': tf.io.VarLenFeature(tf.float32),
        'y_max': tf.io.VarLenFeature(tf.float32),
        'object_name': tf.io.VarLenFeature(tf.string),
        'object_id': tf.io.VarLenFeature(tf.int64)
    }
    return features


def create_example(separate_data, key, image_data):
    """
    Create tf.train.Example object.
    Args:
        separate_data: numpy tensor of 1 image data.
        key: output of hashlib.sha256()
        image_data: raw image data.

    Returns:
        tf.train.Example object.
    """
    [image, object_name, image_width, image_height, x_min, y_min, x_max,
     y_max, _, _, object_id] = separate_data
    image_file_name = os.path.split(image[0])[-1]
    image_format = image_file_name.split('.')[-1]
    features = {
        'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height[0]])),
        'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width[0]])),
        'image_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image[0].encode('utf-8')])),
        'image_file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            image_file_name.encode('utf8')])),
        'image_key': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'image_format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format.encode('utf8')])),
        'x_min': tf.train.Feature(float_list=tf.train.FloatList(value=x_min)),
        'y_min': tf.train.Feature(float_list=tf.train.FloatList(value=y_min)),
        'x_max': tf.train.Feature(float_list=tf.train.FloatList(value=x_max)),
        'y_max': tf.train.Feature(float_list=tf.train.FloatList(value=y_max)),
        'object_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=object_name)),
        'object_id': tf.train.Feature(int64_list=tf.train.Int64List(value=object_id)),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def read_example(example, feature_map, class_table, max_boxes, new_size=None):
    """
    Read single a single example from a TFRecord file.
    Args:
        example: nd tensor.
        feature_map: A dictionary of feature names mapped to tf.io objects.
        class_table: StaticHashTable object.
        max_boxes: Maximum number of boxes per image
        new_size: w, h new image size

    Returns:
        x_train, y_train
    """
    features = tf.io.parse_single_example(example, feature_map)
    x_train = tf.image.decode_png(features['image_data'], channels=3)
    if new_size:
        x_train = tf.image.resize(x_train, new_size)
    object_name = tf.sparse.to_dense(features['object_name'])
    label = tf.cast(class_table.lookup(object_name), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(features[feature])
                        for feature in ['x_min', 'y_min', 'x_max', 'y_max']] + [label], 1)
    padding = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, padding)
    return x_train, y_train


def write_tf_record(output_path, groups, data, trainer=None):
    """
    Write data to TFRecord.
    Args:
        output_path: Full path to save.
        groups: pandas GroupBy object.
        data: pandas DataFrame
        trainer: Main.Trainer object.

    Returns:
        None
    """
    print(f'Processing {os.path.split(output_path)[-1]}')
    if trainer:
        if 'train' in output_path:
            trainer.train_tf_record = output_path
        if 'test' in output_path:
            trainer.valid_tf_record = output_path
    with tf.io.TFRecordWriter(output_path) as r_writer:
        for current_image, (image_path, objects) in enumerate(groups.iteritems(), 1):
            print(f'\rBuilding example: {current_image}/{len(groups)} ... '
                  f'{os.path.split(image_path)[-1]} '
                  f'{round(100 * (current_image / len(groups)))}% completed', end='')
            separate_data = pd.DataFrame(objects, columns=data.columns).T.to_numpy()
            image_width, image_height, x_min, y_min, x_max, y_max = separate_data[2: 8]
            x_min /= image_width
            x_max /= image_width
            y_min /= image_height
            y_max /= image_height
            image_data = open(image_path, 'rb').read()
            key = hashlib.sha256(image_data).hexdigest()
            training_example = create_example(separate_data, key, image_data)
            r_writer.write(training_example.SerializeToString())
    print()


def save_tfr(data, output_folder, dataset_name, test_size=None, trainer=None):
    """
    Transform and save dataset into TFRecord format.
    Args:
        data: pandas DataFrame with the dataset contents.
        output_folder: Path to folder where TFRecord(s) will be saved.
        dataset_name: str name of the dataset.
        test_size: relative test subset size.
        trainer: Main.Trainer object

    Returns:
        None
    """
    data['Object Name'] = data['Object Name'].apply(lambda x: x.encode('utf-8'))
    data['Object ID'] = data['Object ID'].astype(int)
    groups = data.groupby('Image Path').apply(np.array)
    if test_size:
        assert 0 < test_size < 1, f'test_size must be 0 < test_size < 1 and {test_size} is given'
        separation_index = int((1 - test_size) * len(groups))
        training_set = groups[:separation_index]
        test_set = groups[separation_index:]
        training_path = os.path.join(output_folder, f'{dataset_name}_train.tfrecord')
        test_path = os.path.join(output_folder, f'{dataset_name}_test.tfrecord')
        write_tf_record(training_path, training_set, data, trainer)
        write_tf_record(test_path, test_set, data, trainer)
        return
    tf_record_path = os.path.join(output_folder, f'{dataset_name}.tfrecord')
    write_tf_record(tf_record_path, groups, data, trainer)


def read_tfr(tf_record_file, classes_file, feature_map, max_boxes,
             classes_delimiter='\n', new_size=None):
    """
    Read and load dataset from TFRecord file.
    Args:
        tf_record_file: Path to TFRecord file.
        classes_file: file containing classes.
        feature_map: A dictionary of feature names mapped to tf.io objects.
        max_boxes: Maximum number of boxes per image.
        classes_delimiter: delimiter in classes_file.
        new_size: w, h new image size

    Returns:
        MapDataset object.
    """
    text_init = tf.lookup.TextFileInitializer(
        classes_file, tf.string, 0, tf.int64, -1, delimiter=classes_delimiter)
    class_table = tf.lookup.StaticHashTable(text_init, -1)
    files = tf.data.Dataset.list_files(tf_record_file)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(
        lambda x: read_example(x, feature_map, class_table, max_boxes, new_size))


if __name__ == '__main__':
    from Helpers.annotation_parsers import parse_voc_folder
    fr = parse_voc_folder(folder_path='../../../beverly_hills/labels',
                          voc_conf='../Config/voc_conf.json')
    save_tfr(fr, '../../', 'beverly_hills', 0.2)
    # read_tfr('../../beverly_hills_train.tfrecord', '../Config/beverly_hills.txt', get_feature_map(),
    #          100)