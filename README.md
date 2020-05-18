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
## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)

* [Description](#description)

* [Features](#features)
  * [`tensorflow` 2.2 & `keras` functional api.](#tf2)
  * [CPU & GPU support.](#gpu-cpu)
  * [Random weights & DarkNet weights support.](#weights)
  * [csv-xml annotation parsers.](#annot)
  * [anchor generator.](#anchor-gen)
  * [`matplotlib` visualization of all stages.](#visual)
  * [`tf.data` input pipeline.](#input)
  * [`pandas` & `numpy` data handling.](#pn)
  * [`imgaug` augmentation pipeline(customizable).](#aug)
  * [`logging` coverage.](#log)
  * [all-in-1 custom trainer.](#trainer)
  * [Stop and resume training support.](#stop)
  * [fully vectorized mAP evaluation.](#evaluate)
  * [`labelpix` support.](#labelpix)
  * [Photo and video detection.](#photo-vid)

* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

<!-- GETTING STARTED -->
## Getting started

### Prerequisites

Here are the packages you'll need to install before starting to use the detector:
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

### Installation
1. Clone the repo
```sh
git clone https://github.com/emadboctorx/yolov3-keras-tf2/
```
2. Install requirements
```sh
pip install -r requirements.txt
```
or
```sh
conda install --file requirements.txt
```

<!-- DESCRIPTION -->
## Description
yolov3-keras-tf2 is an implementation of [yolov3](https://pjreddie.com/darknet/yolo/) (you only look once)
which is is a state-of-the-art, real-time object detection system that is extremely fast and accurate.
There are many implementations that support tensorflow, only a few that support tensorflow v2 and as I did
not find versions that suit my needs so, I decided to create this version which is very flexible and 
customizable. It requires the Python interpreter version 3.6, 3.7, 3.7+, is not platform specific and is 
MIT licensed which means you can use, copy, modify, distribute this software however you like.













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
