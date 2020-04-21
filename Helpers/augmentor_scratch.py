import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from Config.augmentation_options import augmentations


def aug_sometimes(aug):
    """
    Helper for further augmentations.
    Args:
        aug: augmentation technique.

    Returns:
        ia.Sometimes
    """
    return iaa.Sometimes(0.5, aug)


def ratios_to_coordinates(bx, by, bw, bh, width, height):
    """
    Convert relative coordinates to actual coordinates.
    Args:
        bx: Relative center x coordinate.
        by: Relative center y coordinate.
        bw: Relative box width.
        bh: Relative box height.
        width: Current image display space width.
        height: Current image display space height.

    Return:
        x1: x coordinate.
        y1: y coordinate.
        x2: x1 + Bounding box width.
        y2: y1 + Bounding box height.
    """
    w, h = bw * width, bh * height
    x, y = bx * width + (w / 2), by * height + (h / 2)
    return x, y, x + w, y + h


def convert_relative_coords(labels):
    """
    Convert labels to coordinates and save to csv.
    Args:
        labels: csv label data.

    Returns:
        None
    """
    w, h = 1344, 756
    mapping = pd.read_csv(labels)
    items_to_save = []
    for index, data in mapping.iterrows():
        image_name, object_name, object_index, bx, by, bw, bh = data
        x1, y1, x2, y2 = ratios_to_coordinates(bx, by, bw, bh, w, h)
        items_to_save.append(
            [image_name, x1, y1, x2, y2, object_name, object_index, bx, by, bw, bh])
    new_data = pd.DataFrame(
        items_to_save, columns=['image', 'x1', 'y1', 'x2', 'y2', 'object_type',
                                'object_id', 'bx', 'by', 'bw', 'bh'])
    new_data[['x1', 'y1', 'x2', 'y2']] = new_data[['x1', 'y1', 'x2', 'y2']].astype('int64')
    new_data.to_csv('scratch/label_coordinates.csv', index=False)


def calculate_ratios(x1, y1, x2, y2, width, height):
    """
    Calculate relative object ratios in the labeled image.
    Args:
        x1: Start x coordinate.
        y1: Start y coordinate.
        x2: End x coordinate.
        y2: End y coordinate.
        width: Bounding box width.
        height: Bounding box height.

    Return:
        bx: Relative center x coordinate.
        by: Relative center y coordinate.
        bw: Relative box width.
        bh: Relative box height.
    """
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    bx = 1 - ((width - min(x1, x2) + (box_width / 2)) / width)
    by = 1 - ((height - min(y1, y2) + (box_height / 2)) / height)
    bw = box_width / width
    bh = box_height / height
    return bx, by, bw, bh


def get_bounding_boxes(image_path, csv_data='scratch/label_coordinates.csv'):
    """
    Get all boxes in the given image for displaying purposes.
    Args:
        image_path: Path to image.
        csv_data: csv image label mapping.

    Returns:
        pandas DataFrame with x1, y1, x2, y2 as columns.
    """
    image_name = os.path.basename(image_path)
    data = pd.read_csv(csv_data)
    groups = data.groupby('image')
    return groups.get_group(image_name)


def aug1(images):
    """
    The following example shows a standard use case. An augmentation sequence
    (crop + horizontal flips + gaussian blur) is defined once at the start of
    the script. Then many batches are loaded and augmented before being used
    for training.
    Args:
        images: img tensor.

    Returns:
        img, aug
    """
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])
    img_aug = seq(images=images)
    return images, img_aug


def aug2(images):
    """
    The following example shows an augmentation sequence that might
    be useful for many common experiments. It applies crops and affine
    transformations to images, flips some of the images horizontally,
    adds a bit of noise and blur and also changes the contrast as well
    as brightness.
    Args:
        images: img tensor.

    Returns:
        img, aug
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq(images=images)
    return images, images_aug


def aug3(images):
    """
    The following example shows a large augmentation sequence containing
    many different augmenters, leading to significant changes in the augmented
    images. Depending on the use case, the sequence might be too strong.
    Occasionally it can also break images by changing them too much.
    To weaken the effects you can lower the value of iaa.SomeOf((0, 5), ...)
    to e.g. (0, 3) or decrease the probability of some augmenters to be applied
    by decreasing in sometimes = lambda aug: iaa.Sometimes(0.5, aug) the value 0.5
    to e.g. 0.3.
    Args:
        images: img tensor.

    Returns:
        img, aug
    """
    # aug_sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. aug_sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            aug_sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            aug_sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           aug_sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 1.0),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           aug_sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's channel with 5% probability.
                           # This sets each pixel value v to 255-v.
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           aug_sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           aug_sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    images_aug = seq(images=images)
    return images, images_aug


def get_bounding_boxes_over_image(image, path):
    """
    Get BoundingBoxesOnImage object.
    Args:
        image: single image
        path: single path.

    Returns:
        BoundingBoxesOnImage, frame_before.
    """
    boxes = []
    frame_before = get_bounding_boxes(path)
    for item in frame_before[['x1', 'y1', 'x2', 'y2']].values:
        boxes.append(BoundingBox(*item))
    return BoundingBoxesOnImage(boxes, shape=image.shape[:2]), frame_before


def get_adjusted_frames(bbs_aug, frame_before, image, image_aug, image_before, image_after):
    frame_after = pd.DataFrame(bbs_aug.bounding_boxes, columns=['x1y1', 'x2y2'])
    frame_after = pd.DataFrame(np.hstack((frame_after['x1y1'].tolist(), frame_after['x2y2'].tolist())),
                               columns=['x1', 'y1', 'x2', 'y2']).astype('int64')
    for column in frame_after:
        frame_before[f'{column}_new'] = frame_after[column].values
    return image, image_aug, image_before, image_after, frame_after


def aug4(image, path):
    """
    The following example loads an image and places two bounding boxes on it.
    The image is then augmented to be brighter, slightly rotated and scaled.
    These augmentations are also applied to the bounding boxes. The image is
    then shown before and after augmentation (with bounding boxes drawn on it).
    Args:
        image: single image
        path: single path.

    Returns:
        image, image_aug, image_before, image_after, frame_before
    """
    bbs, frame_before = get_bounding_boxes_over_image(image, path)
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
              )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
    return get_adjusted_frames(bbs_aug, frame_before, image, image_aug, image_before, image_after)


def display_with_bbs(aug, images, image_paths):
    for image, image_path in zip(images, image_paths):
        bbs, frame_before = get_bounding_boxes_over_image(image, image_path)
        image_aug, bbs_aug = aug(image=image, bounding_boxes=bbs)
        image_before = bbs.draw_on_image(image, size=2)
        image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
        cv2.imshow('before', image_before)
        cv2.imshow('after', image_after)
        cv2.waitKey(0)


def aug5(image, path):
    """
    Bounding boxes can easily be projected onto rescaled versions
    of the same image using the function .on(image). This changes the
    coordinates of the bounding boxes. E.g. if the top left coordinate
    of the bounding box was before at x=10% and y=15%, it will still be
    at x/y 10%/15% on the new image, though the absolute pixel values
    will change depending on the height/width of the new image.
    Args:
        image: single image
        path: single path.

    Returns:
        image, image_aug, image_before, image_after, frame_before
    """
    bbs, frame_before = get_bounding_boxes_over_image(image, path)
    # image = ia.quokka(size=(256, 256))
    # bbs = BoundingBoxesOnImage([
    #     BoundingBox(x1=25, x2=75, y1=25, y2=75),
    #     BoundingBox(x1=100, x2=150, y1=25, y2=75)
    # ], shape=image.shape)

    # Rescale image and bounding boxes
    image_rescaled = ia.imresize_single_image(image, (512, 512))
    bbs_rescaled = bbs.on(image_rescaled)

    # Draw image before/after rescaling and with rescaled bounding boxes
    image_bbs = bbs.draw_on_image(image, size=2)
    image_rescaled_bbs = bbs_rescaled.draw_on_image(image_rescaled, size=2)
    return get_adjusted_frames(bbs_rescaled, frame_before, image, image_rescaled,
                               image_bbs, image_rescaled_bbs)


def aug6(image):
    """
    The following example loads a standard image and a generates a
    corresponding heatmap. The heatmap is supposed to be a depth map,
    i.e. is supposed to resemble the depth of objects in the image,
    where higher values indicate that objects are further away.
    (For simplicity we just use a simple gradient as a depth map
    with a cross in the center, so there is no real correspondence
    between the image and the depth values.)
    Args:
        image: single image

    Returns:
        heat map.
    """
    # Create an example depth map (float32, 128x128).
    # Here, we use a simple gradient that has low values (around 0.0)
    # towards the left of the image and high values (around 50.0)
    # towards the right. This is obviously a very unrealistic depth
    # map, but makes the example easier.
    depth = np.linspace(0, 50, 128).astype(np.float32)  # 128 values from 0.0 to 50.0
    depth = np.tile(depth.reshape(1, 128), (128, 1))  # change to a horizontal gradient

    # We add a cross to the center of the depth map, so that we can more
    # easily see the effects of augmentations.
    depth[64 - 2:64 + 2, 16:128 - 16] = 0.75 * 50.0  # line from left to right
    depth[16:128 - 16, 64 - 2:64 + 2] = 1.0 * 50.0  # line from top to bottom

    # Convert our numpy array depth map to a heatmap object.
    # We have to add the shape of the underlying image, as that is necessary
    # for some augmentations.
    depth = HeatmapsOnImage(
        depth, shape=image.shape[:2], min_value=0.0, max_value=50.0)

    # To save some computation time, we want our models to perform downscaling
    # and hence need the ground truth depth maps to be at a resolution of
    # 64x64 instead of the 128x128 of the input image.
    # Here, we use simple average pooling to perform the downscaling.
    depth = depth.avg_pool(2)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),  # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
    ], random_order=True)

    # Augment images and heatmaps.
    images_aug = []
    heatmaps_aug = []
    for _ in range(5):
        images_aug_i, heatmaps_aug_i = seq(image=image, heatmaps=depth)
        images_aug.append(images_aug_i)
        heatmaps_aug.append(heatmaps_aug_i)

    # We want to generate an image of original input images and heatmaps
    # before/after augmentation.
    # It is supposed to have five columns:
    # (1) original image,
    # (2) augmented image,
    # (3) augmented heatmap on top of augmented image,
    # (4) augmented heatmap on its own in jet color map,
    # (5) augmented heatmap on its own in intensity colormap.
    # We now generate the cells of these columns.
    #
    # Note that we add a [0] after each heatmap draw command. That's because
    # the heatmaps object can contain many sub-heatmaps and hence we draw
    # command returns a list of drawn sub-heatmaps.
    # We only used one sub-heatmap, so our lists always have one entry.
    cells = []
    for image_aug, heatmap_aug in zip(images_aug, heatmaps_aug):
        cells.append(image)  # column 1
        cells.append(image_aug)  # column 2
        cells.append(heatmap_aug.draw_on_image(image_aug)[0])  # column 3
        cells.append(heatmap_aug.draw(size=image_aug.shape[:2])[0])  # column 4
        cells.append(heatmap_aug.draw(size=image_aug.shape[:2], cmap=None)[0])  # column 5

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    # imageio.imwrite("example_heatmaps.jpg", grid_image)
    return grid_image


def aug7(images):
    seq = iaa.BlendAlpha(
        factor=(0.2, 0.8),
        foreground=iaa.Sharpen(1.0, lightness=2),
        background=iaa.CoarseDropout(p=0.1, size_px=8)
    )

    images_aug = seq(images=images)
    return images, images_aug


def aug8(images):
    seq = iaa.BlendAlphaFrequencyNoise(
        foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    images_aug = seq.augment_images(images)
    return images, images_aug


def aug9(images):
    seq = iaa.BlendAlphaFrequencyNoise(
        foreground=iaa.EdgeDetect(1.0),
        per_channel=True
    )
    images_aug = seq.augment_images(images)
    return images, images_aug


def aug10(images):
    seq = iaa.BlendAlphaFrequencyNoise(
        exponent=-2,
        foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        size_px_max=32,
        upscale_method="linear",
        iterations=1,
        sigmoid=False
    )
    images_aug = seq.augment_images(images)
    return images, images_aug


def aug11(images):
    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2 * 255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    images_aug = aug.augment_images(images)
    return images, images_aug


def aug12(images):
    aug = iaa.SomeOf((0, None), [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2 * 255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    images_aug = aug.augment_images(images)
    return images, images_aug


def aug13(images):
    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2 * 255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ], random_order=True)
    images_aug = aug.augment_images(images)
    return images, images_aug


def display_augmentations(img_tensor, images_aug):
    for img, img_aug in zip(img_tensor, images_aug):
        new_real_size = (np.array(img.shape[:2]) * 0.7).astype(np.int64)
        new_aug_size = (np.array(img_aug.shape[:2]) * 0.7).astype(np.int64)
        new_real = cv2.resize(img, tuple(reversed(new_real_size)))
        new_aug = cv2.resize(img_aug, tuple(reversed(new_aug_size)))
        cv2.imshow('Real', new_real)
        cv2.imshow('Aug', new_aug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_augmentations(images, group, bbs=False, paths=None):
    """
    Test augmentations from augmentation_options.
    Args:
        images: image tensor
        group: One of the options.
        bbs: if True, bounding boxes will be displayed over images.
        paths: image paths.

    Returns:
        None
    """
    for augmentation in group:
        try:
            aug = eval(augmentation['augmentation'])
            print(augmentation['no'])
            if not bbs and not paths:
                img_aug = aug(images=images)
                display_augmentations(images, img_aug)
            else:
                display_with_bbs(aug, images, paths)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    imgs = [Path(f'../../../beverly_hills/photos/{photo}').absolute().resolve()
            for photo in os.listdir('../../../beverly_hills/photos/')][:2]
    dat = [cv2.imread(f'{image}') for image in imgs]
    img_tens = np.array(dat)
    aug4(img_tens[0], imgs[0])
    # test_augmentations(img_tens, augmentations['size'], True, imgs)
    # im, ima = aug3(img_tens)
    # display_augmentations(im, ima)
    # convert_relative_coords(labels='../../../beverly_hills/bh_labels.csv')
    # label_path = 'scratch/label_coordinates.csv'
    # print(get_bounding_boxes('Beverly_hills100.png', label_path))