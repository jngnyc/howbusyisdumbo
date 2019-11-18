import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from time import sleep
import cv2
from picamera.array import PiRGBArray
import picamera
import yaml

from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import datetime
from pytz import timezone
from twitter_utils import _post_update

from twython import Twython
from twython import TwythonStreamer


image_number = 0
config = []

TESTING_LOCALLY = False

IMAGE_RAW = (os.getcwd() + "/images/" + ('raw%s.jpg' % 1))
IMAGE_BOUNDED = (os.getcwd() + "/images/" + ('bounded%s.jpg' % 1))
IMAGE_TO_TWITTER_RAW = 'twitterimage.jpg'
IMAGE_TO_TWITTER_BOUNDED = 'twitterimage_bounded.jpg'
 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' #fast
MODEL_FILE = MODEL_NAME + '.tar.gz' 
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb' 
PATH_TO_LABELS = '/home/pi/howbusyisdumbo/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

def _init_inference_graph():
    print('initializing inference graph...')
    
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
                
    print('done!')
                
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
            
    print('done!')
            
    return detection_graph, category_index
            
            
def _init_camera_interface():
    print('initializing camera...')
    camera = picamera.PiCamera()
    #camera.resolution = (1920, 1088)
    camera.resolution = (config['width'], config['height'])
    camera.sharpness = 100
    camera.iso = 200
    camera.framerate = 30
    camera.exif_tags['IFD0.Artist'] = 'twitter.com/howbusyisdumbo'
    camera.exif_tags['IFD0.Copyright'] = 'Copyright (c) 2019 twitter.com/howbusyisdumbo'
    sleep(2)
    
    return camera
    
    
# Setup callbacks from Twython Streamer
class BlinkyStreamer(TwythonStreamer):
        def on_success(self, data):
                #print data
                if 'text' in data:
                        twitter = Twython(
                            config['app_key'],
                            config['app_secret'],
                            config['oauth_token'],
                            config['oauth_token_secret'] )                  
                        twitter.update_status(status='@{twtr_id} say cheese!'.format(twtr_id = data['user']['screen_name'].encode('utf-8') ))
                        camera = _init_camera_interface()
                        camera.capture('/home/pi/Desktop/image.jpg')
                        photo = open('/home/pi/Desktop/image.jpg', 'rb')
                        response = twitter.upload_media(media = photo)
                        twitter.update_status(status='@{twtr_id}'.format(twtr_id = data['user']['screen_name'].encode('utf-8') ), media_ids = [response['media_id']])
                        
                        camera.close()


def _detection_loop(detection_graph, category_index, camera, loop_num):

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess: 
            
            # init camera
            rawCapture = picamera.array.PiRGBArray(camera, size = ((1920, 1080) ) )
            
            # take a picture
            camera.capture(rawCapture, 'bgr')
        
            image_np = np.array(rawCapture.array)                                  
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

            (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})

            # count number of objects in the image
            numPeople = 0
            for index, value in enumerate(classes[0]):
                if ( (scores[0, index] > config['min_confidence']) & (value == 1) ):
                    numPeople = numPeople + 1
                        
                        
            cv2.imwrite( (os.getcwd() + "/images/" + ('raw%s.jpg' % loop_num)), image_np)
            
            # only visualize boxes around people, ignore everything else
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            ind = np.argwhere(classes == 1)
            boxes = np.squeeze(boxes[ind])
            scores = np.squeeze(scores[ind])
            classes = np.squeeze(classes[ind])

            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, 
                    boxes,
                    classes.astype(np.int32),
                    scores,
                    category_index,
                    use_normalized_coordinates = True,
                    min_score_thresh =  config['min_confidence'],
                    line_thickness = 2) 

            # don't waste cycles if we don't need to display it
            if (config['visualize_obj_detection'] == True):
                cv2.imshow('object detection', cv2.resize(image_np, (1280, 960)))
                
            cv2.imwrite( (os.getcwd() + "/images/" + ('bounded%s.jpg' % loop_num)), image_np)
            
            return numPeople
            
            
class testobj(object):
    def __init__(self, file_raw = None, file_bounded = None, num_people = 0, time = None):
        self.file_raw = file_raw
        self.file_bounded = file_bounded
        self.num_people = num_people
        self.time = time
    
# MAIN START
def main():
    
    print('\n****\nConfiguration:')
    print( '- testing locally? {p}'.format(p = TESTING_LOCALLY) )
    print( '- how many pictures to check before declaring a max = {p}'.format(p = config['how_many_to_check']) )
    print( '- post to Twitter? {p}'.format(p = ("True" if (TESTING_LOCALLY == False) else "False")) )
    print( '- visualize to display? {p}'.format(p = config['visualize_obj_detection'])    )
    print( '- minimum confidence level = {c}%'.format(c = (config['min_confidence'] * 100)) )
    print( '****\n')

    _init_inference_graph()
    detection_graph, category_index = _init_detection_graph()
    camera = _init_camera_interface()
    count = 1
    currentMax = 0
    candidate_pool = []
    
    print('starting main detection loop')
    
    while True:

        # TODO - make timezone a parameter in config.yml
        ts = datetime.datetime.now(timezone('EST'))
        start = time.time()
        
        numPeople = _detection_loop(detection_graph, category_index, camera, count)        

        # only bother if we detect at least 1 person
        if(numPeople):
            print( ("{:d}/{:d} - detected {:d} " + ("person" if (numPeople == 1) else "people") + " at " + ((ts.strftime("%H:%M")))).format(count, config['how_many_to_check'], numPeople) )
            
            candidate_pool.append(testobj((os.getcwd() + "/images/" + ('raw%s.jpg' % count)),
              (os.getcwd() + "/images/" + ('bounded%s.jpg' % count)),
              numPeople,
              (ts.strftime("%H:%M"))))
        
        # done looking for people! now lets find the picture that had the most people in it
        if (count == config['how_many_to_check']):
            for each in candidate_pool:
                if (each.num_people > currentMax):
                    # new max found
                    print "new max found - num_people: {:d}! raw:{:s} - time: {}".format(each.num_people, each.file_raw, each.time)
                    currentMax = each.num_people
                    maxPic = each

            # post to Twitter (if not running locally)
            if (TESTING_LOCALLY != True):
                try:
                    _post_update(maxPic.num_people, maxPic.file_raw, maxPic.file_bounded)
                except NameError:
                    pass

            # reset the counters
            count = 0
            currentMax = 0
            candidate_pool = []
            print "done finding a max in a pool of {:s} images - starting a new candidate pool of images now...".format(str(config['how_many_to_check']))
        
        end = time.time()
        print "loop {:d} took {:.1f}s".format(count, end - start)
        count += 1
            
        # cleanup
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            camera.close()
            break
    


if __name__== "__main__":
    
    config = yaml.safe_load(open(os.path.join(sys.path[0], "config.yml")))
    #image_number = 0
    
    for i in sys.argv[1:]:
        if (i == "localtest"):
            print("localtest")
            TESTING_LOCALLY = True
    
    main()    
    
    
    
    
    
    
