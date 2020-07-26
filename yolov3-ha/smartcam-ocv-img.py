#
# Example of using keras-yolo3 for detecting object on a camera.
# The idea is to detect persons or moving objects and to build a
# warning/notification system for that.
#
# Also allow plugin of Home-automation integration and calls:
#
# - ha_detect.publish_detection(detect_type, max_score) - top detection
# - ha_detect.publish_image(png-image) - detection image with boxes
#
#
# Author: Joakim Eriksson, joakim.eriksson@ri.se
#

import cv2, numpy as np, datetime
from PIL import Image, ImageFont, ImageDraw
import sys, importlib, getopt, yaml
import os.path
import colorsys
import hacv
import yolo3

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

def usage():
        print("Usage: ", sys.argv[0],"[-v <URI>] [-s] [-d]")
        print("Options:")
        print("-h             help - show this info")
        print("-v <URI>       fetch video from this URI")
        print("-p <pkg.class> plugin for the video detection notifications")
        print("-s             show input and detections (openCV)")
        print("-d             save detections to disk")
        print("-c             load config file")

def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)/new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)/new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

yolo = yolo3.YoloV3(confThreshold, nmsThreshold)

video_path = "0"
show = False
save_to_disk = False
plugin = "hacv.CVMQTTPlugin".split(".")
config = None
yaml_cfg = None

try:
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"hsdv:p:c:")
except getopt.GetoptError as e:
    sys.stderr.write(str(e) + '\n')
    usage()
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        usage()
        sys.exit()
    elif opt == "-s":
        show = True
    elif opt == "-d":
        save_to_disk = True
    elif opt == "-v":
        video_path = arg
    elif opt == "-p":
        plugin = arg.split(".")
    elif opt == "-c":
        config = arg

if config is not None:
        with open(config, 'r') as ymlfile:
                yaml_cfg = yaml.load(ymlfile)
        print("Config: ", yaml_cfg)
        cvconf = yaml_cfg['cvconf']
        plugin = cvconf['plugin'].split(".")
        if video_path == 0:
            video_path = cvconf['video']

# allow video_path "0" => first camera (web-camera on my Macbook)
if video_path == "0":
   video_path = 0

# load image
image = cv2.imread(video_path)
image_h, image_w, _ = image.shape
net_h, net_w = 416, 416
new_image = preprocess_input(image, net_h, net_w)

# create the plugin
cls = getattr(importlib.import_module(plugin[0]), plugin[1])
ha_detect = cls(yaml_cfg)

detection = yolo.detect(image)

# Put efficiency information. The function getPerfProfile returns the
# overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = yolo.net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
# Write the frame with the detection boxes
if len(detection) > 0:
    max_score = detection[0][1]
    detect_name = detection[0][0]
else:
    max_score = 0
print("current max score %d", max_score)
# only publish if score is higher than zero
if max_score > 0:
    print("*** Detected ", detect_name)
    ha_detect.publish_detection(detect_name, max_score)
    ha_detect.publish_detections(detection)
    ha_detect.publish_image(cv2.imencode('.png', image)[1].tostring())
    ha_detect.__del__()
# show the image and save detection disk
if show:
    cv2.imshow("YOLOv3", image)
if save_to_disk:
    file = 'yolo-' + detect_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
    cv2.imwrite(file, image)

sys.exit(0)
