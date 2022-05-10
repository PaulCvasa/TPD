import os
import tarfile
import time
import urllib.request
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.1/bin")
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np


DATA_DIR = os.path.join(os.getcwd(), 'G:/tpd_v4/tensorflow/data')
MODELS_DIR = os.path.join(DATA_DIR, 'G:/tpd_v4/tensorflow/models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

# Download and extract model
MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

# Download labels file
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')

# Load the model
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

#Load label map data for plotting
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# Set input
videoInput = cv2.VideoCapture('testRecs/test1.mp4')

# Detection
last_time = time.time()
while True:
    # Read frame from camera
    success, frame = videoInput.read()
    print('loop took {} seconds'.format(time.time() - last_time))
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(frame, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    boxes, scores, classes = detect_fn(input_tensor)

    label_id_offset = 1
    frame_with_detections = frame.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          frame_with_detections,
          boxes['detection_boxes'][0].numpy(),
          (boxes['detection_classes'][0].numpy() + label_id_offset).astype(int),
          boxes['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=20,
          min_score_thresh=.50,
          agnostic_mode=False)

    #print(boxes)

    # Compute distances to TPs and send warnings
    # for i, b in enumerate(boxes['detection_boxes'][0]):
    #     if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:  # if it's a car, a truck or a bus
    #         if scores[0][i] > 0.5:  # if the confidence level is bigger than 50%
    #             mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
    #             mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
    #             aproxDist = round((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4, 1)  # the power can be tweaked,
    #                                                                                 # to affect granularity
    #             cv2.putText(frame, '{}'.format(aproxDist), (int(mid_x * 1300), int(mid_y * 550)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #             #           image         text                       text position                text font            size     color   line width
    #             if aproxDist <= 0.5:
    #                 if 0.3 < mid_x < 0.7:  # if our object is in our path
    #                     cv2.putText(frame, 'WARNING!', (int(mid_x * 800) + 200, int(mid_y * 450) + 150),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    last_time = time.time()
    cv2.imshow('Traffic Participants Detection', cv2.resize(frame_with_detections, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

videoInput.release()
cv2.destroyAllWindows()