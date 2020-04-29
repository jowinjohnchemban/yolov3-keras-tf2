import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imagesize
import cv2
import os


def save_fig(title, save_figures):
    """
    Save generated figures to Caches folder.
    Args:
        title: Figure title also the image to save file name.
        save_figures: If True, figure will be saved

    Returns:
        None
    """
    if save_figures:
        saving_path = os.path.join('..', 'Caches', f'{title}.png')
        plt.savefig(saving_path)


def visualize_box_relative_sizes(frame, save_result=False):
    """
    Scatter plot annotation box relative sizes.
    Args:
        frame: pandas DataFrame with the annotation data.
        save_result: If True, figure will be saved

    Returns:
        None
    """
    title = f'Relative width and height for {frame.shape[0]} boxes.'
    if os.path.join('..', 'Caches', f'{title}.png') in os.listdir(os.path.join('..', 'Caches')) or (
            frame is None):
        return
    sns.scatterplot(x=frame["Relative Width"], y=frame["Relative Height"], hue=frame["Object Name"],
                    palette='gist_rainbow')
    plt.title(title)
    save_fig(title, save_result)


def visualize_k_means_output(centroids, frame, save_result=False):
    """
    Visualize centroids and anchor box dimensions calculated.
    Args:
        centroids: 2D array of shape(k, 2) output of k-means.
        frame: pandas DataFrame with the annotation data.
        save_result: If True, figure will be saved

    Returns:
        None
    """
    title = f'{centroids.shape[0]} Centroids representing relative anchor sizes.'
    if os.path.join('..', 'Caches', f'{title}.png') in os.listdir(os.path.join('..', 'Caches')) or (
            frame is None):
        return
    fig, ax = plt.subplots()
    visualize_box_relative_sizes(frame)
    plt.title(title)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')
    save_fig(title, save_result)


def visualize_boxes(relative_anchors, sample_image, save_result=False):
    """
    Visualize anchor boxes output of k-means.
    Args:
        relative_anchors: Output of k-means.
        sample_image: Path to image to display as background.
        save_result: If True, figure will be saved

    Returns:
        None
    """
    title = 'Generated anchors relative to sample image size'
    if os.path.join('..', 'Caches', f'{title}.png') in os.listdir(os.path.join('..', 'Caches')):
        return
    img = cv2.imread(sample_image)
    width, height = imagesize.get(sample_image)
    center = int(width / 2), int(height / 2)
    for relative_w, relative_h in relative_anchors:
        box_width = relative_w * width
        box_height = relative_h * height
        x0 = int(center[0] - (box_width / 2))
        y0 = int(center[1] - (box_height / 2))
        x1 = int(x0 + box_width)
        y1 = int(y0 + box_height)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 4)
    plt.imshow(img)
    plt.grid()
    plt.title(title)
    save_fig(title, save_result)


def visualization_wrapper(to_visualize):
    """
    Wrapper for visualization.
    Args:
        to_visualize: function to visualize.

    Returns:
        to_visualize
    """
    def visualized(*args, **kwargs):
        result = to_visualize(*args, **kwargs)
        if to_visualize.__name__ in ['parse_voc_folder', 'adjust_non_voc_csv']:
            visualize_box_relative_sizes(result)
            plt.show()
        if to_visualize.__name__ == 'k_means':
            all_args = list(kwargs.values()) + list(args)
            if not any([isinstance(item, pd.DataFrame) for item in all_args]):
                return result
            visualize_k_means_output(*result)
            plt.show()
            visualize_boxes(result[0], '../sample_image.png')
            plt.show()
        return result
    return visualized





