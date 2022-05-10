import time
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.1/bin")
import tensorflow as tf
import cv2

videoInput = cv2.VideoCapture('testRecs/test1.mp4')

# ## Object detection imports
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# # Detection
gpuOptions = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
last_time = time.time()
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpuOptions)) as sess:
        while videoInput.isOpened():
            success, frame = videoInput.read()
            print('loop took {} seconds'.format(time.time() - last_time))
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            expandedFrame = np.expand_dims(frame, axis=0)
            tensorFrame = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={tensorFrame: expandedFrame})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            # Compute distances to TPs and send warnings
            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:  # if it's a car, a truck or a bus
                    if scores[0][i] > 0.5:  # if the confidence level is bigger than 50%
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        aproxDist = round((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4, 1)  # the power can be tweaked,
                        # to affect granularity
                        cv2.putText(frame, '{}'.format(aproxDist), (int(mid_x * 1300), int(mid_y * 550)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        #           image         text                       text position                text font            size     color   line width
                        if aproxDist <= 0.45:
                            if 0.3 < mid_x < 0.7:  # if our object is in our path
                                cv2.putText(frame, 'WARNING!', (int(mid_x * 800) + 200, int(mid_y * 450) + 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            last_time = time.time()
            cv2.imshow('Traffic Participants Detection', cv2.resize(frame, (800, 450)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
