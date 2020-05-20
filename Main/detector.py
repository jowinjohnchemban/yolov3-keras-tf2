import numpy as np
import cv2
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from Main.models import V3Model
from Helpers.utils import get_detection_data, activate_gpu, transform_images, default_logger, timer


class Detector(V3Model):
    def __init__(self,
                 input_shape,
                 classes_file,
                 anchors=None,
                 masks=None,
                 max_boxes=100,
                 iou_threshold=0.5,
                 score_threshold=0.5):
        self.class_names = [
            item.strip() for item in open(classes_file).readlines()
        ]
        self.box_colors = {class_name: color for class_name, color in zip(
            self.class_names, [list(np.random.random(size=3) * 256)
                               for _ in range(len(self.class_names))]
        )}
        super().__init__(input_shape=input_shape,
                         classes=len(self.class_names),
                         anchors=anchors,
                         masks=masks,
                         max_boxes=max_boxes,
                         iou_threshold=iou_threshold,
                         score_threshold=score_threshold)
        activate_gpu()

    def detect_image(self, image_data, image_name=''):
        image = tf.expand_dims(image_data, 0)
        resized = transform_images(image, self.input_shape[0])
        out = self.inference_model.predict(resized)
        adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
        detections = get_detection_data(
            adjusted,
            image_name,
            out,
            self.class_names,
        )
        return detections, adjusted

    def draw_on_image(self, adjusted, detections):
        for index, row in detections.iterrows():
            img, obj, x1, y1, x2, y2, score, *_ = row.values
            color = self.box_colors.get(obj)
            cv2.rectangle(
                adjusted,
                (x1, y1),
                (x2, y2),
                color,
                2)
            cv2.putText(
                adjusted,
                f'{obj}-{round(score, 2)}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.6,
                color,
                1)

    def predict_on_image(self, image_path):
        image_name = os.path.basename(image_path)
        image_data = tf.image.decode_image(
            open(image_path, 'rb').read(), channels=3)
        detections, adjusted = self.detect_image(image_data, image_name)
        self.draw_on_image(adjusted, detections)
        saving_path = os.path.join('..', 'Output', 'Detections', f'predicted-{image_name}')
        cv2.imwrite(saving_path, adjusted)

    @timer(default_logger)
    def predict_photos(self, photos, trained_weights, batch_size=32, workers=16):
        self.create_models()
        self.load_weights(trained_weights)
        to_predict = photos.copy()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            predicted = 1
            done = []
            total_photos = len(photos)
            while to_predict:
                current_batch = [to_predict.pop() for _ in range(batch_size) if to_predict]
                future_predictions = {executor.submit(self.predict_on_image, image): image
                                      for image in current_batch}
                for future_prediction in as_completed(future_predictions):
                    future_prediction.result()
                    completed = f'{predicted}/{total_photos}'
                    current_image = future_predictions[future_prediction]
                    percent = (predicted / total_photos) * 100
                    print(
                        f'\rpredicting {os.path.basename(current_image)} '
                        f'{completed}\t{percent}% completed',
                        end='',
                    )
                    predicted += 1
                    done.append(current_image)
            for item in done:
                default_logger.info(f'Saved prediction: {item}')


if __name__ == '__main__':
    anc = np.array(
        [[43, 70],
         [82, 66],
         [70, 70],
         [103, 66],
         [107, 66],
         [110, 84],
         [207, 176],
         [178, 414],
         [450, 243]]
    )
    p = Detector(
        (416, 416, 3),
        '../Config/coco.names',
        score_threshold=0.5,
        iou_threshold=0.5,
        max_boxes=100,
        # anchors=anc
                  )
    pred_dir = '/Users/emadboctor/Desktop/beverly_hills/photos/'
    photos_to_predict = [f'{pred_dir}{photo}' for photo in os.listdir(pred_dir)][20:40]
    p.predict_photos(photos_to_predict,
                     '/Users/emadboctor/Desktop/yolov3.weights')#beverly_hills/models/beverly_hills_model.tf')


# from Helpers.utils import ratios_to_coordinates
# import pandas as pd
#
# x = [0.441441441, 0.437888199, 0.154154154, 0.232919255]
# xx1, yy1, xx2, yy2 = ratios_to_coordinates(*x, 1344, 756)
# print([xx2 - xx1, yy2 - yy1])
# z = []
# for ind, r in pd.read_csv('../../../kos.csv').iterrows():
#     img, obj, objin, bx, by, bw, bh = r.values
#     xxx1, yyy1, xxx2, yyy2 = ratios_to_coordinates(bx, by, bw, bh, 1344, 756)
#     z.append([int(xxx2 - xxx1), int(yyy2 - yyy1)])
# print(sorted(z, key=lambda x: x[0] * x[1]))
