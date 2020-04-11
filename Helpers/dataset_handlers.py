import tensorflow as tf
import pandas as pd
import numpy as np
import hashlib
import os


def create_example(separate_data, key, image_data):
    """
    Create tf.train.Example objects
    Args:
        separate_data: numpy tensor of 1 image data.
        key: output of hashlib.sha256()
        image_data: raw image data.

    Returns:
        tf.train.Example object.
    """
    [image, object_names, image_widths, image_heights, x_mins, y_mins, x_maxs,
     y_maxs, bw, bh, object_ids] = separate_data
    image_file_name = os.path.split(image[0])[-1]
    image_format = image_file_name.split('.')[-1]
    features = {
        'image_heights': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_heights[0]])),
        'image_widths': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_widths[0]])),
        'image_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image[0].encode('utf-8')])),
        'image_file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            image_file_name.encode('utf8')])),
        'image_key': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'image_format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format.encode('utf8')])),
        'x_mins': tf.train.Feature(float_list=tf.train.FloatList(value=x_mins)),
        'y_mins': tf.train.Feature(float_list=tf.train.FloatList(value=y_mins)),
        'x_maxs': tf.train.Feature(float_list=tf.train.FloatList(value=x_maxs)),
        'y_maxs': tf.train.Feature(float_list=tf.train.FloatList(value=y_maxs)),
        'object_names': tf.train.Feature(bytes_list=tf.train.BytesList(value=object_names)),
        'object_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=object_ids)),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def save_tfr(data, output_path):
    """
    Transform, split and save dataset into TFRecord format.
    Args:
        data: pandas DataFrame with the dataset contents.
        output_path: Full path/name to save train/validation TFRecords.

    Returns:
        None
    """
    data['Object Name'] = data['Object Name'].apply(lambda x: x.encode('utf-8'))
    data['Object ID'] = data['Object ID'].astype(int)
    groups = data.groupby('Image Path').apply(np.array)
    with tf.io.TFRecordWriter(output_path) as r_writer:
        for image_path, objects in groups.iteritems():
            separate_data = pd.DataFrame(objects, columns=data.columns).T.to_numpy()
            image_widths, image_heights, x_mins, y_mins, x_maxs, y_maxs = separate_data[2:8]
            x_mins /= image_widths
            x_maxs /= image_widths
            y_mins /= image_heights
            y_maxs /= image_heights
            image_data = open(image_path, 'rb').read()
            key = hashlib.sha256(image_data).hexdigest()
            training_example = create_example(separate_data, key, image_data)
            r_writer.write(training_example.SerializeToString())


if __name__ == '__main__':
    from annotation_parsers import parse_voc_folder
    fr = pd.read_csv('../Caches/data_set_labels.csv')
    save_tfr(fr, 'beverly_hills.tfrecord')