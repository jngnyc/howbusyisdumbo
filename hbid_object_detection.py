import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import cv2

from picamera.array import PiRGBArray

import picamera

import logging

from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import datetime
from twitter_utils import postUpdate

VISUALIZE_OBJ_DETECTION = False


RAW_IMAGE = 'raw.jpg'
BOUNDED_IMAGE = 'bounded.jpg'
IMAGE_TO_TWITTER_RAW = 'twitterimage.jpg'
IMAGE_TO_TWITTER_BOUNDED = 'twitterimage_bounded.jpg'
PERIOD = 2     # how often (min) to upload a new image

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' #fast 
#MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017' #medium speed
MODEL_FILE = MODEL_NAME + '.tar.gz' 
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' 
PATH_TO_LABELS = '/home/pi/howbusyisdumbo/tensorflowmodels/models/research/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90
MIN_CONFIDENCE = .4
IMAGE_SIZE = (12,8)

def jimmy():
    print "hello"

def _analyze_or_not():

    # only record during daytime b/c the camera doesn't support nightvision
    hour = (datetime.datetime.today()).hour
    start_time = 6
    end_time = 21
    
    if( (hour < end_time) and (hour >= start_time) ):
        analyze = True
    else:
        analyze = False
    return analyze

fileAlreadyExists = os.path.isfile(PATH_TO_CKPT)

if not fileAlreadyExists:
    print('Downloading frozen inference graph') 
    opener = urllib.request.URLopener() 
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE) 
    tar_file = tarfile.open(MODEL_FILE) 
    for file in tar_file.getmembers(): 
        file_name = os.path.basename(file.name) 
        if 'frozen_inference_graph.pb' in file_name: 
            tar_file.extract(file, os.getcwd())

print('initializing detection graph...')
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

logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

print('initializing camera...')
camera = picamera.PiCamera() 
camera.resolution = (1280, 960) 
#camera.vflip = True
camera.framerate = 30
rawCapture = PiRGBArray(camera, size = (1280, 960))
print('camera init complete!')

print ('starting main detection loop')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess: 
        for frame in camera.capture_continuous(rawCapture, format="bgr"): 
            
            ts = datetime.datetime.today()
            
            image_np = np.array(frame.array)                                  
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3] 
            image_np_expanded = np.expand_dims(image_np, axis=0) 
            # Definite input and output Tensors for detection_graph 
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') 
                                                                                                                         
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
            
            # Each score represent how level of confidence for each of the objects. 
            # Score is shown on the result image, together with the class label. 
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            logging.info('Running detection..')
            
            (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded}) 

            # count number of objects in the image
            numPeople = numCars = 0
            for index, value in enumerate(classes[0]):
                if scores[0, index] > MIN_CONFIDENCE:
                    if(value == 1):
                        numPeople = numPeople + 1
                    #if(value == 3):
                        #numCars = numCars + 1
            
            # every __min period, find the frame that has the most people in it within 1 minute
            if (ts.minute == PERIOD):
                print("within correct period! starting to find max_people_num...")
                if (numPeople > currentMax):
                    print 'new max found: ' + numPeople + ' (previousMax = ' + currentMax + ')'
                    currentMax = numPeople
                    cv2.imwrite(IMAGE_TO_TWITTER_RAW, image_np)
                    write_bounded_img = True
            else:
                print("not within the correct period so I'm not figuring out the max...")
            
            #print( "people: " + str(numPeople) + " - " + "cars: " + str(numCars) )

            logging.info('Done.  Visualizing.. ') 
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, 
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores), 
                    category_index, 
                    use_normalized_coordinates = True,
                    min_score_thresh =  MIN_CONFIDENCE,
                    line_thickness = 3) 
            
            # don't waste cycles if we don't need to display it
            if (VISUALIZE_OBJ_DETECTION == True):
                cv2.imshow('object detection', cv2.resize(image_np, (1280, 960)))
            
            # only write the bounded image if we detected it as the max
            if (write_bounded_img == True):
                cv2.imwrite(IMAGE_TO_TWITTER_BOUNDED, image_np)
                write_bounded_image = False    
            
            # hacky way of determining that we're done  
            # done with finding the max raw/bounded images for this __min period, upload them to twitter
            if (ts.minute == (PERIOD + 1) ):
                postUpdate(currentMax, IMAGE_TO_TWITTER_RAW, IMAGE_TO_TWITTER_BOUNDED)
                currentMax = 0
            
            rawCapture.truncate(0) 
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows() 
                break
        
        print('exiting')
        cap.release() 
        cv2.destroyAllWindows() 
        
        
        
# def _graveyard_functions()
            # upload the raw file to S3
            #cv2.imwrite(RAW_IMAGE, image_np)
            #upload(datetime.datetime.today(), RAW_IMAGE, True, testpath)
            #upload(datetime.datetime.today(), cv2.imencode('.jpg', image_np)[1].tostring(), image_np)

            #mongodb_data = '{"timestamp":"' + str(datetime.datetime.today()) + '", "cars":' + str(numCars) + ', "people":"' + str(numPeople) + '}'
            #logging.debug(json.dumps(mongodb_data))

