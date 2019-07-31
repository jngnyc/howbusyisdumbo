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
from twitter_utils import _post_update


PERIOD = 5    # how often (min) to upload a new image
VISUALIZE_OBJ_DETECTION = False
UPLOAD_TO_TWITTER = True
MIN_CONFIDENCE = .3


RAW_IMAGE = 'raw.jpg'
BOUNDED_IMAGE = 'bounded.jpg'
IMAGE_TO_TWITTER_RAW = 'twitterimage.jpg'
IMAGE_TO_TWITTER_BOUNDED = 'twitterimage_bounded.jpg'


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' #fast 
#MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017' #medium speed
MODEL_FILE = MODEL_NAME + '.tar.gz' 
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' 
PATH_TO_LABELS = '/home/pi/howbusyisdumbo/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

IMAGE_SIZE = (12,8)

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
    
def _init_inference_graph():
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
                
def _init_detection_graph():
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
            
    return detection_graph, category_index
            
            
def _init_camera_interface():
    print('initializing camera...')
    camera = picamera.PiCamera() 
    camera.resolution = (2592, 1952)
    #camera.vflip = True
    camera.framerate = 15
    rawCapture = PiRGBArray(camera, size = (2592, 1952) )
    print('camera init complete!')
    
    return camera, rawCapture
    
    
def filter_boxes(boxes, scores, classes, categories):
    n = len(classes)
    print('catgories = ' + str(categories) )
    print( 'len(classes) = {l}'.format(l = n ))
    idxs = []
    for i in range(n):
        print('classes[i] = {a}'.format(a = classes[i]) )
        if classes[i] in categories:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    
    print ('filtered_classes: ' + str(filtered_classes) )
    
    return filtered_boxes, filtered_scores, filtered_classes
    
    
    
# MAIN START
def main():
    print('\n****Configuration:')
    print( '- post to Twitter? {p}'.format(p = UPLOAD_TO_TWITTER) )
    print( '- post to Twitter period = {p}m'.format(p = PERIOD) )
    print( '- visualize to display? {p}'.format(p = VISUALIZE_OBJ_DETECTION)    )
    print( '- minimum confidence level = {c}%'.format(c = (MIN_CONFIDENCE * 100)) )
    print( '****\n')

    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    _init_inference_graph()

    detection_graph, category_index = _init_detection_graph()

    camera, rawCapture = _init_camera_interface()    
    
    print ('starting main detection loop')
    
    
    #init vars
    current_max = 0
    write_bounded_img = False
    posted_to_twitter = False

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
            
                #(boxes, scores, classes, num) = sess.run(
                #        [detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded}) 

                # count number of objects in the image
                numPeople = numCars = 0
                for index, value in enumerate(classes[0]):
                    if scores[0, index] > MIN_CONFIDENCE:
                        if(value == 1):
                            numPeople = numPeople + 1
                        #if(value == 3):
                            #numCars = numCars + 1
            
                # every __min period, find the frame that has the most people in it within 1 minute
                if ( (ts.minute % PERIOD) == 0 ):
                    print( "ts.minute = {m} - finding max_people_num...".format(m = ts.minute) )
                    posted_to_twitter = False
                    if (numPeople > current_max):
                        print 'new max found: ' + str(numPeople) + ' (previousMax = ' + str(current_max) + ')'
                        current_max = numPeople
                        cv2.imwrite(IMAGE_TO_TWITTER_RAW, image_np)
                        write_bounded_img = True
                else:
                    print("...")
            
                #print( "people: " + str(numPeople) + " - " + "cars: " + str(numCars) )

                logging.info('Done.  Visualizing.. ')
            
                # TODO - remove cars from being visualized
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np, 
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates = True,
                        min_score_thresh =  MIN_CONFIDENCE,
                        line_thickness = 2) 
            
                # don't waste cycles if we don't need to display it
                if (VISUALIZE_OBJ_DETECTION == True):
                    cv2.imshow('object detection', cv2.resize(image_np, (1280, 960)))
            
                # only write the bounded image if we detected it as the max
                if (write_bounded_img == True):
                    cv2.imwrite(IMAGE_TO_TWITTER_BOUNDED, image_np)
                    write_bounded_img = False    
            
                # hacky way of determining that we're done  
                # done with finding the max raw/bounded images for this __min period, upload them to twitter
                if ( (((ts.minute + 1) % PERIOD) == 0) and (posted_to_twitter == False) ):
                    print "I think I'm all done, uploading to twitter and resetting..."
                
                    if( UPLOAD_TO_TWITTER == True):
                        _post_update(current_max, IMAGE_TO_TWITTER_RAW, IMAGE_TO_TWITTER_BOUNDED)
                        posted_to_twitter = True
                        current_max = 0
                    elif (posted_to_twitter == True):
                        print "already posted to twitter..."
            
                rawCapture.truncate(0) 
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    break
        
            print('exiting')
            cap.release() 
            cv2.destroyAllWindows()     
    
  
if __name__== "__main__":

    print ('Command line arguments:', str(sys.argv))
    print ('bye bye')


    #main()    








