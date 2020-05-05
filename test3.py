import pandas as pd
import numpy as np
import os


for item in os.listdir('Caches'):
    if item.endswith('csv'):
        data = pd.read_csv(f'Caches/{item}')
        # data['true_positive'] = np.nan
        # data['false_positive'] = np.nan
        # data['false_negative'] = np.nan
        # data = data[['image', 'object_name', 'x1', 'y1', 'x2', 'y2', 'score', 'true_positive',
        #              'false_positive', 'false_negative', 'image_width', 'image_height']]
        data['max_iou'] = 0
        data.to_csv(f'Caches/{item}', index=False)