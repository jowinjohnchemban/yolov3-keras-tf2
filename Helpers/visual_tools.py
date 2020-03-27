import matplotlib.pyplot as plt
from annotation_parsers import parse_voc_folder
import seaborn as sns


def visualize_box_relative_sizes(folder_path, voc_conf, cache_file='data_set_labels.csv'):
    """
    Scatter plot annotation box relative sizes.
    Args:
        folder_path: Path to folder containing xml annotations.
        voc_conf: Path to voc json configuration file.
        cache_file: csv file name containing current session labels.

    Return:
        None
    """
    frame = parse_voc_folder(folder_path, voc_conf, cache_file)
    title = f'Relative width and height for {frame.shape[0]} boxes.'
    sns.scatterplot(x=frame["Relative Width"], y=frame["Relative Height"], hue=frame["Object Name"],
                    palette='gist_rainbow')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    visualize_box_relative_sizes('../../../beverly_hills_gcp/lbl', '../Config/voc_conf.json', 'data_set_labels.csv')