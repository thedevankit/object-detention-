import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import requests
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import smtplib

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("0")
	# out = cv2.VideoWriter('outpt.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 1, (640,480))

sys.path.append("..")

from  object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

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



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
cntr = 0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # small_frame = cv2.resize(frame2, (640, 480), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      # image_np = small_frame[:, :, ::-1]


      # for video out put befor processing 
      # frame = cv2.resize(image_np, (640,480))
      # out.write(frame)


      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      class_for_lbl = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, class_for_lbl, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      count=np.squeeze(scores)
      cnt=0
      for i in range(100):
        if scores is None or count[i]>0.5:
          cnt= cnt+1
        print("Total Object Detected")
        print(cnt)


 # for simple mail opration  it send a mail to user for detected img

        # newclasses = []
        # with open('classes_int.txt', 'a') as file:
        # 	print(classes, file=file)
        # for c in classes:
        #     newclasses.append(int(c))


# it send a message to user for detected img 


        # if 1 in classes:
        #   print("Bottle Found",cntr)
        #   cntr +=1
        #   if cntr == 200:
        #     print("send SMS")
        #     print("Bottle Found")
        #     cntr =0
        #     apikey='E3wxF3vTcOA-6I5WHNYxX6SWSLRDa5Qo63n5YWVHgY'
        #     numbers=['7798210599']
        #     sender='NOTYBX'
        #     template='Camera Notification: '
        #     sms='Human Action Detected'
        #     message=template+sms

        #     payload={'apiKey':apikey,'numbers':numbers,'sender':sender,'message':message}
        #     response = requests.post('http://api.textlocal.in/send/',data=payload,timeout=5)
        #     print(response.text)

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)        
      
      cv2.imshow('object detection', cv2.resize(image_np,(800,600)))
      if 1.0 in classes:
      	cntr+=1
      	if cntr==10:
      		print("Human action found")
      		cv2.imwrite("ankit.jpg",image_np)
      		cntr=0


      # def printit():
      # 	threading.Timer(5.0, printit).start()
      	
      	# frame = cv2.resize(image_np, (640,480))
      	# out.write(frame)
      # 	cv2.imwrite("test.jpg",image_np)
      # 	print ("Hello, World!")
      # printit() 

# save detected img to disk

      # for frm in image_np:
      #   cntr +=1
      #   if cntr==400:
      #       cv2.imwrite("test.jpg",image_np)
      #       print("saved")
      #       cntr=0

            
# send img with attachment of img 

            # from email.mime.text import MIMEText
            # from email.mime.image import MIMEImage
            # from email.mime.multipart import MIMEMultipart
            # fromaddr = 'wcyber23@gmail.com'
            # toaddrs  = 'Ankitk.as51@gmail.com'
            # username = 'wcyber23@gmail.com'
            # password = 'ANKITK.AS51'
            # server = smtplib.SMTP('smtp.gmail.com:587')
            # server.ehlo()
            # msg = MIMEMultipart()
            # fp = open('ankit.jpg', 'rb')
            # img = MIMEImage(fp.read())
            # fp.close()
            # msg.attach(img)
            # server.starttls()
            # server.login(username,password)
            # server.sendmail(fromaddr, toaddrs, msg.as_string())
            # print('sent')
            # server.quit()

      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

