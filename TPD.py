import time
import threading
import numpy as np
import os

# Paths to the CUDA Toolkit and CUDNN library
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
#os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.1/bin")

import tensorflow as tf
import cv2
import PySimpleGUI as psg
# Object detection imports from the Tensorflow module
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from CameraFrameGetter import *

def modelSetup():
    # # Model preparation
    # ## Variables
    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FDG` to point to a new .pb file.
    # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FDG = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

    # Number of object classes (eg. car, truck etc.)
    NUM_CLASSES = 8

    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_FDG, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`,
    # we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index


def computeDistanceAndSendWarning(frame, boxes, classes, scores, roadType):
    for i, b in enumerate(boxes[0]):  # iterate through detection boxes
        if scores[0][i] > 0.4:  # if the confidence level is bigger than 50%
            # we subtract the width difference from 1, so the value will be towards 0 when the object is closer and to 1 when it's far away
            aproxDist = round((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 8, 2) * 20  # distance computed using detection box width, multiplied by 20 to convert in meters
                                                                                     # the power can be tweaked to affect granularity
            middleX = (boxes[0][i][1] + boxes[0][i][3]) / 2  # middle of the X coord of detected object
            middleY = (boxes[0][i][0] + boxes[0][i][2]) / 2  # middle of the Y coord of detected object

            # if it's a bicycle,       a car,              a motorcycle,           a bus,              a train                 or a truck
            if classes[0][i] == 2 or classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 6 or classes[0][i] == 7 or classes[0][i] == 8:
                #           image         text                          text position                     text font            size       color   line width
                cv2.putText(frame, '{:0.1f} m'.format(aproxDist), (int(middleX * 1300), int(middleY * 550)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show aproximate distance
                if roadType == 'highway':      # verify type of road
                    if 0.45 < middleX < 0.55:    # if the object is in the ego vehicle path
                        if aproxDist <= 13:  # if the aproximate distance is smaller than the threshold
                            cv2.putText(frame, '!WARNING!', (int(middleX * 1280), int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
                if roadType == 'normal':       # verify type of road
                    if 0.4 < middleX < 0.6:      # if the object is in the ego vehicle path
                        if aproxDist <= 9:  # if the aproximate distance is smaller than the threshold
                            cv2.putText(frame, '!WARNING!', (int(middleX * 1280), int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
                if roadType == 'city':         # verify type of road
                    if 0.3 < middleX < 0.7:      # if the object is in the ego vehicle path
                        if aproxDist < 6:   # if the aproximate distance is smaller than the threshold
                            cv2.putText(frame, '!WARNING!', (int(middleX * 1280), int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)  # send warning
            elif classes[0][i] == 1 :  # if it's a pedestrian
                #           image           text                                  text position                 text font         size       color   line width
                cv2.putText(frame, '{:0.1f} m'.format((aproxDist)/3), (int(middleX * 1300), int(middleY * 550)), cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)  # show aproximate distance
                if 0.2 < middleX < 0.8:  # if the object is in the ego vehicle path
                    cv2.putText(frame, '!WARNING!', (int(middleX * 1280) , int(middleY * 720)), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 3)

# Different method used for the webcam input, because of the CameraFrameGetter optimization
def detection(input, roadType):
    # Initialize model
    detection_graph, category_index = modelSetup()
    # GPU memory % allocated
    gpuOptions = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
    # Variable for last frame processed time
    lastFrameTime = 0
    # Variable for current frame processed time
    currentFrameTime = 0
    with detection_graph.as_default():
        # Start TensorFlow session
        with tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpuOptions)) as sess:
            while True:
                frame = input.read()
                # print('loop took {} seconds'.format(time.time() - last_time))
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                expandedFrame = np.expand_dims(frame, axis=0) # axis = 0 -> rows; axis = 1 -> columns
                # Tensor created for the frame
                tensorFrame = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                tensorBoxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represents the level of confidence for each of the detected objects.
                # Score is shown on the result image, together with the class label.
                tensorScores = detection_graph.get_tensor_by_name('detection_scores:0')
                tensorClasses = detection_graph.get_tensor_by_name('detection_classes:0')
                # Tensor for the number of detections
                tensorNum_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (tensorBoxes, tensorScores, tensorClasses, tensorNum_detections) = sess.run(
                    [tensorBoxes, tensorScores, tensorClasses, tensorNum_detections],
                    feed_dict={tensorFrame: expandedFrame})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(tensorBoxes),
                    np.squeeze(tensorClasses).astype(np.int32),
                    np.squeeze(tensorScores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=.4,
                    line_thickness=4)

                # Compute distances to TPs and send warnings
                t = threading.Thread(target=computeDistanceAndSendWarning, args=[frame, tensorBoxes, tensorClasses, tensorScores, roadType])
                t.start()
                t.join()

                currentFrameTime = time.time()
                fps = str(int(1 / (currentFrameTime - lastFrameTime)))
                lastFrameTime = currentFrameTime

                cv2.putText(frame, fps, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Traffic Participants Detection - Press Q to stop the detection',
                            cv2.resize(frame, (950, 600)))

                if cv2.waitKey(25) & 0xFF == ord('q') or input.stopped:
                    input.stop()
                    cv2.destroyAllWindows()
                    break

def testDetection(input, roadType):
    # Initialize model
    detection_graph, category_index = modelSetup()
    # GPU memory % allocated
    gpuOptions = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
    # Variable for last frame processed time
    lastFrameTime = 0
    # Variable for current frame processed time
    currentFrameTime = 0
    with detection_graph.as_default():
        # Start TensorFlow session
        with tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpuOptions)) as sess:
            while True:
                success, frame = input.read()
                while success:
                    # print('loop took {} seconds'.format(time.time() - last_time))
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    expandedFrame = np.expand_dims(frame, axis=0) # axis = 0 -> rows; axis = 1 -> columns
                    # Tensor created for the frame
                    tensorFrame = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    tensorBoxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represents the level of confidence for each of the detected objects.
                    # Score is shown on the result image, together with the class label.
                    tensorScores = detection_graph.get_tensor_by_name('detection_scores:0')
                    tensorClasses = detection_graph.get_tensor_by_name('detection_classes:0')
                    # Tensor for number of detections
                    tensorNum_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (tensorBoxes, tensorScores, tensorClasses, tensorNum_detections) = sess.run(
                        [tensorBoxes, tensorScores, tensorClasses, tensorNum_detections],
                        feed_dict={tensorFrame: expandedFrame})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(tensorBoxes),
                        np.squeeze(tensorClasses).astype(np.int32),
                        np.squeeze(tensorScores),
                        category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=.4,
                        line_thickness=4)

                    # Compute distances to TPs and send warnings
                    t = threading.Thread(target=computeDistanceAndSendWarning, args=[frame, tensorBoxes, tensorClasses, tensorScores, roadType])
                    t.start()
                    t.join()

                    currentFrameTime = time.time()
                    fps = str(int(1 / (currentFrameTime - lastFrameTime)))
                    lastFrameTime = currentFrameTime

                    cv2.putText(frame, fps, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow('Traffic Participants Detection - Press Q to stop the detection',
                               cv2.resize(frame, (950, 600)))

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    success, frame = input.read()
                break
            input.release()
            cv2.destroyAllWindows()


def main():
    roadType = 'normal'

    psg.theme("Dark")
    mainFont = "Comic Sans MS", "10"
    secondaryFont = "Comic Sans MS", "14"
    # Main buttons layout
    mainButtonsLine = [
        [psg.Button("START DETECTION", size=(20, 7), font=mainFont),
         psg.Button("TEST DETECTION", size=(20, 7), button_color='Blue', font=mainFont),
         psg.Button("EXIT", size=(20, 7), button_color='Red', font=mainFont)]
    ]

    # Settings Buttons layout
    settingsButtonsLine = [
        [psg.Button("CITY ROAD", size=(20, 7), button_color='Gray', font=mainFont, key="city"),
         psg.Button("NORMAL ROAD", size=(20, 7), button_color='Black', font=mainFont, key="normal"),
         psg.Button("HIGHWAY", size=(20, 7), button_color='Gray', font=mainFont, key="highway")]
    ]

    # Main layout window
    mainLayout = [
        [psg.Text("TRAFFIC PARTICIPANTS DETECTION", size=(120,0), justification="center", font=("Comic Sans MS", "20"))],
        [psg.Text("CURRENT ROAD TYPE SETTING: ", size=(30, 25), font=secondaryFont), psg.Text("normal", size=(60, 25), key='setting', font=secondaryFont)],
        [psg.Column(mainButtonsLine, vertical_alignment='center', justification='center', k='-C-')],
        [psg.Column(settingsButtonsLine, vertical_alignment='center', justification='center', k='-C-')]
    ]

    mainWindow = psg.Window("TRAFFIC PARTICIPANTS DETECTION", mainLayout, resizable=True, finalize=True)
    mainWindow.Maximize()
    while True:
        event, values = mainWindow.read(timeout=20)
        if event == "EXIT" or event == psg.WIN_CLOSED:
            break
        elif event == "START DETECTION":
            print(roadType)
            videoInput = CameraFrameGetter(0).start()
            detection(videoInput, roadType)
        elif event == "TEST DETECTION":
            print(roadType)
            testInput = cv2.VideoCapture('testRecs/night2.mp4')
            testDetection(testInput, roadType)
        elif event == "city":
            roadType = "city"
            mainWindow['setting'].update(value=roadType)
            mainWindow["city"].Update(button_color=('Black'))
            mainWindow["normal"].Update(button_color=('Grey'))
            mainWindow["highway"].Update(button_color=('Grey'))
        elif event == "normal":
            roadType = "normal"
            mainWindow['setting'].update(value=roadType)
            mainWindow["normal"].Update(button_color=('Black'))
            mainWindow["city"].Update(button_color=('Grey'))
            mainWindow["highway"].Update(button_color=('Grey'))
        elif event == "highway":
            roadType = "highway"
            mainWindow['setting'].update(value=roadType)
            mainWindow["highway"].Update(button_color=('Black'))
            mainWindow["normal"].Update(button_color=('Grey'))
            mainWindow["city"].Update(button_color=('Grey'))


main()
