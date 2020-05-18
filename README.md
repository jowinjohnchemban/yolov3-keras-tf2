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
  * [`tf.data` input pipeline.](#tf.data)
  * [`pandas` & `numpy` data handling.](#pn)
  * [`imgaug` augmentation pipeline(customizable).](#aug)
  * [`logging` coverage.](#log)
  * [All-in-1 custom trainer.](#trainer)
  * [Stop and resume training support.](#stop)
  * [Fully vectorized mAP evaluation.](#evaluate)
  * [`labelpix` support.](#labelpix)
  * [Photo and video detection.](#photo-vid)

* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

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

* **Dataset pre and post augmentation visualization with bounding boxes:**

You can always visualize different stages of the program using my other repo 
[labelpix](https://github.com/emadboctorx/labelpix) which is tool for drawing 
bounding boxes, but can also be used to visualize bounding boxes over images using 
csv files in the format mentioned [here](#csv-xml-annotation-parsers).































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
