import tensorflow as tf
from utils import backbone
from api import object_counting_api

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = "./input_footage/trim.mp4"
# input_video="rtsp://admin:admin@555@192.168.1.108/cam/realmonitor?channel=1&subtype=1"

detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

fps = 80
width = 1550
height = 1028
is_color_recognition_enabled = 0
roi = 430
deviation = 10

object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation) # counting all the objects
