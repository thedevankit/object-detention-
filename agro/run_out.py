import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
cap = cv2.VideoCapture(0)

sys.path.append("..")

from  object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
# retrain data set
PATH_TO_CKPT ='retrained_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'retrained_labels.txt'

NUM_CLASSES = 2

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)
frame_count = 0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      frame_count +=1
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      #image_np_expanded = np.expand_dims(image_np, axis=0)
      #image_data = tf.gfile.FastGFile(image_np_expanded, 'rb').read()
      #image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      if frame_count%5 == 0:
        cv2.imwrite("current_frame.jpg",image_np)
        image_data = tf.gfile.FastGFile("current_frame.jpg", 'rb').read()

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
      
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
          human_string = label_lines[node_id]
          score = predictions[0][node_id]
          
          if score>0.40:
            print('%s (score = %.5f)' % (human_string, score))


        # Each box represents a part of the image where a particular object was detected.

        # boxes = detection_graph.get_tensor_by_name('final_result:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        # scores = detection_graph.get_tensor_by_name('final_result:0')
        # classes = detection_graph.get_tensor_by_name('final_result:0')
        # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # # Actual detection.
        # (scores, classes, num_detections) = sess.run(
        #     [scores, classes, num_detections],
        #     feed_dict={image_tensor: image_np_expanded})
        # # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image_np,
        #     np.squeeze(classes).astype(np.int32),
        #     np.squeeze(scores),
        #     category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=8)

      cv2.imshow('object Identification', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
