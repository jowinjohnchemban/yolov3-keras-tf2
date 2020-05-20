[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<p>
  <a href="https://github.com/emadboctorx/yolov3-keras-tf2/">
  </a>

  <h3 align="center">YoloV3 Real Time Object Detector in tensorflow 2.2</h3>
    .
    <a href="https://github.com/emadboctorx/yolov3-keras-tf2/tree/master/Docs"><strong>Explore the docs »</strong></a>
    ·
    <a href="https://github.com/emadboctorx/yolov3-keras-tf2/issues">Report Bug</a>
    ·
    <a href="https://github.com/emadboctorx/yolov3-keras-tf2/issues">Request Feature</a>
  </p>

<!-- TABLE OF CONTENTS -->
## **Table of Contents**

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)

* [Description](#description)

* [Features](#features)
  * [tensorflow-2.X--keras-functional-api](#tensorflow-22--keras-functional-api)
  * [cpu-gpu support](#cpu--gpu-support)
  * [Random weights and DarkNet weights support](#random-weights-and-darknet-weights-support)
  * [csv-xml annotation parsers.](#csv-xml-annotation-parsers)
  * [Anchor generator.](#anchor-generator)
  * [`matplotlib` visualization of all stages.](#matplotlib-visualization-of-all-stages)
  * [`tf.data` input pipeline.](#tfdata-input-pipeline)
  * [`pandas` & `numpy` data handling.](#pandas--numpy-data-handling)
  * [`imgaug` augmentation pipeline(customizable).](#imgaug-augmentation-pipelinecustomizable)
  * [`logging` coverage.](#logging)
  * [All-in-1 custom trainer.](#all-in-1-custom-trainer-class)
  * [Stop and resume training support.](#stop-and-resume-training-support)
  * [Fully vectorized mAP evaluation.](#fully-vectorized-map-evaluation)
  * [`labelpix` support.](#labelpix-support)
  * [Photo & video detection](#photo--video-detection)

* [Usage](#usage)
  * [Training](#training)
  * [Augmentation](#augmentation)
  * [Evaluation](#evaluation)
  * [Detection](#detection)
* [Contributing](#contributing)
* [License](#license)
* [Show your support](#show-your-support)
* [Contact](#contact)

![GitHub Logo](/Samples/detections.png)

<!-- GETTING STARTED -->
## **Getting started**

### **Prerequisites**

Here are the **packages** you'll need to install before starting to use the detector:
* pandas==1.0.3
* lxml==4.5.0
* opencv_python_headless==4.2.0.34
* imagesize==1.2.0
* seaborn==0.10.0
* tensorflow==2.2.0
* tensorflow-gpu==2.2.0
* numpy==1.18.2
* matplotlib==3.2.1
* imgaug==0.4.0

### **Installation**

1. **Clone the repo**
```sh
git clone https://github.com/emadboctorx/yolov3-keras-tf2/
```
2. **Install requirements**
```sh
pip install -r requirements.txt
```
**or**
```sh
conda install --file requirements.txt
```

<!-- DESCRIPTION -->
## **Description**

yolov3-keras-tf2 is an implementation of [yolov3](https://pjreddie.com/darknet/yolo/) (you only look once)
which is is a state-of-the-art, real-time object detection system that is extremely fast and accurate.
There are many implementations that support tensorflow, only a few that support tensorflow v2 and as I did
not find versions that suit my needs so, I decided to create this version which is very flexible and 
customizable. It requires the Python interpreter version 3.6, 3.7, 3.7+, is not platform specific and is 
MIT licensed which means you can use, copy, modify, distribute this software however you like.

<!-- FEATURES -->
## **Features**

### **tensorflow 2.2 & keras functional api**

This program leverages features that were introduced in tensorflow 2.0 
including: 

* **Eager execution:** an imperative programming environment that evaluates operations immediately,
 without building graphs check [here](https://www.tensorflow.org/guide/eager)
* **`tf.function`:** A JIT compilation decorator that speeds up some components of the program check [here](https://www.tensorflow.org/api_docs/python/tf/function)
* **`tf.data`:** API for input pipelines check [here](https://www.tensorflow.org/guide/data)

### **CPU & GPU support**

The program detects and uses available GPUs at runtime(training/detection)
if no GPUs available, the CPU will be used(slow).

### **Random weights and DarkNet weights support**

Both options are available, and NOTE in case of using DarkNet [yolov3 weights](https://pjreddie.com/media/files/yolov3.weights)
you must maintain the same number of [COCO classes](https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda) (80 classes)
as transfer learning to models with different classes will be supported in future versions of this program.

### **csv-xml annotation parsers**

There are 2 currently supported formats that the program is able to read and translate to input.

* **XML VOC format which looks like the following example:**

```xml
<annotation>
	<folder>/path/to/image/folder</folder>
	<filename>image_filename.png</filename>
	<path>/path/to/image/folder/image_filename.png</path>
	<size>
		<width>image_width</width>
		<height>image_height</height>
		<depth>image_depth</depth>
	</size>
	<object>
		<name>obj1_name</name>
		<bndbox>
			<xmin>382.99999987200005</xmin>
			<ymin>447.000000174</ymin>
			<xmax>400.00000051200004</xmax>
			<ymax>469.000000098</ymax>
		</bndbox>
</annotation>
```

* **CSV with relative labels that looks like the following example:**

Image | Object Name | Object Index | bx | by | bw | bh | #
--- | --- | --- | --- |--- |--- |--- |--- 
img1.png | dog | 2 | 0.438616071 | 0.51521164 | 0.079613095	| 0.123015873
img1.png | car | 1 | 0.177827381 | 0.381613757 | 0.044642857 | 0.091269841
img2.png | Street Sign | 5 | 0.674107143 | 0.44047619 | 0.040178571 | 0.084656085

### **Anchor generator**

A [k-means](https://en.wikipedia.org/wiki/K-means_clustering) algorithm finds the optimal sizes and generates 
anchors with process visualization.

### **matplotlib visualization of all stages**

**Including:**

* **k-means visualization:**

![GitHub Logo](/Samples/anchors.png)

* **Generated anchors:**

![GitHub Logo](/Samples/anchors_sample.png)

* **Precision and recall curves:**

![GitHub Logo](/Samples/pr.png)

* **Evaluation bar charts:**

![GitHub Logo](/Samples/map.png)

* **Actual vs. detections:**

![GitHub Logo](/Samples/true_false.png)

* **Augmentation options visualization:**

Double screen visualization(before/after) image like the following example:

![GitHub Logo](/Samples/aug1.png)

* **Dataset pre and post augmentation visualization with bounding boxes:**

You can always visualize different stages of the program using my other repo 
[labelpix](https://github.com/emadboctorx/labelpix) which is tool for drawing 
bounding boxes, but can also be used to visualize bounding boxes over images using 
csv files in the format mentioned [here](#csv-xml-annotation-parsers).

### **`tf.data` input pipeline**

[TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) a simple format for storing a sequence 
of binary records. Protocol buffers are a cross-platform, cross-language library for efficient serialization of 
structured data and are used as input pipeline to store and read data efficiently
the program takes as input images and their respective annotations and builds training and validation(optional)
TFRecords to be further used for all operations and TFRecords are also used in the evaluation(mid/post) training,
so it's valid to say you can delete images to free space after conversion to TFRecords.

### **`pandas` & `numpy` data handling**

Most of the operations are using numpy and pandas for efficiency and vectorization.

### **`imgaug` augmentation pipeline(customizable)**

Special thanks to the amazing [imgaug](https://github.com/aleju/imgaug) creators,
an augmentation pipeline(optional) is available and NOTE that the augmentation is
conducted **before** the training not during the training due to technical complications
to integrate tensorflow and imgaug. If you have a small dataset, augmentation is an option
and it can be preconfigured before the training check [Augmentor.md](Docs/Augmentor.md)

### **`logging`**

Different operations are recorded using `logging` module.

### **All-in-1 custom `Trainer` class**

For custom training, `Trainer` class accepts configurations for augmentation, 
new anchor generation, new dataset(TFRecord(s)) creation, mAP evaluation
mid-training and post training. So all you have to do is place images
in Data > Photos, provide the configuration that suits you and start the training
process, all operations are managed from the same place for convenience.
For detailed instructions check [Trainer.md](Docs/Trainer.md)

### **Stop and resume training support**

by default the trainer checkpoints to Models > checkpoint_name.tf at the end
of each training epoch which enables the training to be resumed at any given 
point by loading the checkpoint which would be the most recent.

### **Fully vectorized mAP evaluation**

Evaluation is optional during the training every n epochs(not recommended for 
large datasets as it predicts every image in the dataset) and one evaluation 
at the end which is optional as well. Training and validation datasets
can be evaluated separately and calculate mAP(mean average precision) as well
as precision and recall curves for every class in the model, 
check [Evaluator.md](Docs/Evaluator.md)

### **labelpix support**

You can check my other repo [labelpix](https://github.com/emadboctorx/labelpix) which is a
labeling tool for drawing bounding boxes over images if you need to make custom datasets
the tool can help and is supported by the detector. You can use csv files
in the format mentioned [here](#csv-xml-annotation-parsers) as labels and load
images if you need to preview any stage of the training/augmentation/evaluation/detection.

### **Photo & video detection**

Detections can be performed on photos or videos using Predictor class
check [Predictor.md](/Docs/Predictor.md)

## **Usage**

### **Training**

**Here are the most basic steps to train using a custom dataset:**

1- Copy images to Data > Photos

2- If labels are in the XML VOC [format](#csv-xml-annotation-parsers),
copy label xml files to Data > Labels

3- Create classes .txt file that contains classes delimited by \n


    dog
    cat
    car
    person
    boat
    fan
    laptop


4- Create a training instance and specify `input_shape`, `classes_file`,
`image_width` and `image_height`


    trainer = Trainer(
             input_shape=(416, 416, 3),
             classes_file='/path/to/classes_file.txt',
             image_width=1344,  # The original image width
             image_height=756   # The original image height
    )

5- Create dataset configuration(dict) that contains the following keys:

- `dataset_name`: TFRecord prefix(required)

and one of the following:(required)

- `relative_labels`: path to csv file in the following [format](#csv-xml-annotation-parsers)

or

- `from_xml`: `True` 

and

- `test_size`: percentage of the validation split ex: 0.1(optional)
- `augmentation`: `True` (optional)

and if `augmentation` this implies the following:

- `sequences`: (required) A list of augmentation sequences check [Augmentor.md](Docs/Augmentor.md) 
- `workers`: (optional) defaults to 32 parallel augmentations.
- `batch_size`: (optional) this is the augmentation batch size defaults to 64 images to load at once.

      dataset_conf = {
                    'relative_labels': '/path/to/labels.csv',
                    'dataset_name': 'dataset_name',
                    'test_size': 0.2,
                    'sequences': preset_1,  # check Config > augmentation_options.py
                    'augmentation': True,
      }

6- Create new anchor generation configuration(dict) that contains the following keys:

- `anchors_no`: number of anchors(should be 9)
and one of the following:
    -  `relative_labels`: same as dataset configuration above
    - `from_xml`: same as dataset configuration above
    
          anchors_conf = {
                          'anchors_no': 9,
                          'relative_labels':  '/path/to/labels.csv'
          }

7- Start the training

**Note** 

If you're going to use DarkNet yolov3 weights, make sure the classes file
contains 80 classes(COCO classes) or you'll get an error. Transfer learning 
to models with different number of classes will be supported in future versions
of the program.

    tr.train(epochs=100, 
             batch_size=8, 
             learning_rate=1e-3, 
             dataset_name='dataset_name', 
             merge_evaluation=False,
             min_overlaps=0.5,
             new_dataset_conf=dataset_conf,  # check step 5
             new_anchors_conf=anchors_conf,  # check step 6
             #  weights='/path/to/weights'  # If you're using DarkNet weights or resuming training
             )
             

After the training completes:

1. The trained model is saved in Models folder(which you can use to resume training later/predict photos or videos)
2. The resulting TFRecords and their corresponding csv data are saved in Data > TFRecords
3. The resulting figures and evaluation results are saved in Output folder.

### **Augmentation**

**Here are the most basic steps to augment images(no training, just augmentation):**

If you need to augment photos and take your time to examine/visualize the results,
here are the steps:

1- Copy images to Data > Photos or specify `image_folder` param

2- Ensure you have a csv file containing the labels in the format 
mentioned [here](#csv-xml-annotation-parsers), if you have labels
in xml VOC format, you can easily convert them using Helpers > annotation_parsers.py > 
`parse_voc_folder()` (everything is explained in the docstrings)

3- Create augmentation instance:

    from Config.augmentation_options import augmentations
    from Helpers.augmentor import DataAugment
    
    
    aug = DataAugment(
          labels_file='/path/to/labels/csv/file',
          augmentation_map=augmentations)
    aug.create_sequences(sequences)  # check the docs
    aug.augment_photo_folder()

After augmentation you'll find augmented images in the Data > Photos folder
or the folder you specified(if you did specify one) 

And you should find 2 csv files in the Output folder: 

1. `augmented_data_plus_original.csv` : you can use this with 
[labelpix](https://github.com/emadboctorx/labelpix) to visualize results with
bounding boxes

2. `adjusted_data_plus_original.csv`

and any of the 2 csv files above can be used in the new dataset configuration
in the training.

## **Evaluation**

Here are the most basic steps to evaluate a trained model:

1. Create an evaluation instance:

       evaluator = Evaluator(
                   input_shape=(416, 416, 3),
                   train_tf_record='/path/to/train.tfrecord',
                   valid_tf_record='/path/to/valid.tfrecord',
                   classes_file='/path/to/classes.txt',
                   anchors=anchors,  # defaults to yolov3 anchors
                   score_threshold=0.1  # defaults to 0.5 but it's okay to be lower
                   )
                   
2. Read actual and prediction results(that resulted from the training)

       actual = pd.read_csv('../Data/TFRecords/full_data.csv')
       preds = pd.read_csv('../Output/full_dataset_predictions.csv')
       
3. Calculate mAP(mean average precision):

       evaluator.calculate_map(
                  prediction_data=preds, 
                  actual_data=actual, 
                  min_overlaps=0.5, 
                  display_stats=True)

After evaluation, you'll find resulting plots and predictions in the Output folder.

### **Detection**

Here are the most basic steps to perform detection:

1. Create an evaluation instance:

        p = Detector(
            (416, 416, 3),
            '/path/to/classes_file.txt',
            score_threshold=0.5,
            iou_threshold=0.5,
            max_boxes=100,
            anchors=anchors  # Optional if not specified, yolo default anchors are used
        )
2. Perform detections:

A) Photos:

    photos = ['photo/path1', 'photo/path2']
    p.predict_photos(photos=photos,
                     trained_weights='/path/to/trained/weights')  # .tf or yolov3.weights(80 classes)

B) Video

    p.detect_video(
        '/path/to/target/vid',
        '/path/to/trained/weights.tf',
    )

After predictions is complete you'll find photos/video
 in Output > Detections

## **Contributing**

Contributions are what make the open source community such an amazing place to  
learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## **License**

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## **Show your support**

Give a ⭐️ if this project helped you!

## **Contact**

Emad Boctor - emad_1989@hotmail.com

Project link: https://github.com/emadboctorx/yolov3-keras-tf2
                   


[contributors-shield]: https://img.shields.io/github/contributors/emadboctorx/yolov3-keras-tf2?style=flat-square
[contributors-url]: https://github.com/emadboctorx/yolov3-keras-tf2/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/emadboctorx/yolov3-keras-tf2?style=flat-square
[forks-url]: https://github.com/emadboctorx/yolov3-keras-tf2/network/members
[stars-shield]: https://img.shields.io/github/stars/emadboctorx/yolov3-keras-tf2?style=flat-square
[stars-url]: https://github.com/emadboctorx/yolov3-keras-tf2/stargazers
[issues-shield]: https://img.shields.io/github/issues/emadboctorx/yolov3-keras-tf2?style=flat-square
[issues-url]: https://github.com/emadboctorx/yolov3-keras-tf2/issues
[license-shield]: https://img.shields.io/github/license/emadboctorx/yolov3-keras-tf2
[license-url]: https://github.com/emadboctorx/yolov3-keras-tf2/blob/master/LICENSE
