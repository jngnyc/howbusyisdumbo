# howbusyisdumbo

I count people taking selfies of my window in DUMBO, Brooklyn using [Tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) running locally on a Raspberry Pi then I post them to Twitter.

Check out how well I count at [@howbusyisdumbo](twitter.com/howbusyisdumbo)

![18 people](https://pbs.twimg.com/media/D-q75VNXsAAuubl?format=jpg&name=large)

## Instructions
- 1. [Install Tensorflow on your Raspberry Pi](https://www.tensorflow.org/install/source_rpi)
- 2. [Download Tensorflow's Mobilnet detector trained on COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) 
- 3. [Insert your own Twitter app keys]()
- 4. Run it - `python mainloop.py`
