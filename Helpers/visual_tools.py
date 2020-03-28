import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imagesize
import cv2


def visualize_box_relative_sizes(frame):
    """
    Scatter plot annotation box relative sizes.
    Args:
        frame: pandas DataFrame with the annotation data.

    Return:
        None
    """
    title = f'Relative width and height for {frame.shape[0]} boxes.'
    sns.scatterplot(x=frame["Relative Width"], y=frame["Relative Height"], hue=frame["Object Name"],
                    palette='gist_rainbow')
    plt.title(title)


def visualize_k_means_output(centroids, frame):
    """
    Visualize centroids and anchor box dimensions calculated.
    Args:
        centroids: 2D array of shape(k, 2) output of k-means.
        frame: pandas DataFrame with the annotation data.

    Return:
        None
    """
    fig, ax = plt.subplots()
    visualize_box_relative_sizes(frame)
    plt.title(f'{centroids.shape[0]} Centroids representing relative anchor sizes.')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')


def visualize_boxes(relative_anchors, sample_image):
    """
    Visualize anchor boxes output of k-means.
    Args:
        relative_anchors: Output of k-means.
        sample_image: Path to image to display as background.

    Return:
        None
    """
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
    plt.title('Generated anchors relative to sample image size')


def visualization_wrapper(to_visualize):
    """
    Wrapper for visualization.
    Args:
        to_visualize: function to visualize.

    Return:
        to_visualize
    """
    def visualized(*args, **kwargs):
        result = to_visualize(*args, **kwargs)
        if to_visualize.__name__ == 'parse_voc_folder':
            visualize_box_relative_sizes(result)
            plt.show()
        if to_visualize.__name__ == 'k_means':
            all_args = list(kwargs.values()) + list(args)
            if not any([isinstance(item, pd.DataFrame) for item in all_args]):
                return result
            visualize_k_means_output(*result)
            plt.show()
            visualize_boxes(result[0], '../sample_img.png')
            plt.show()
        return result
    return visualized





