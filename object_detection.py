#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Object detection
Make Vector dectect objects.
"""
import io
import os
import sys
import time
import random
import math
import argparse
import numpy as np
import tensorflow as tf

try:
    from PIL import Image
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

# Imports the Google Cloud client library
# from google.cloud import vision
# from google.cloud.vision import types

# Imports the Anki Vector SDK
import anki_vector
from anki_vector.util import degrees, distance_mm, speed_mmps

# Imports the TensorFlow Image Labeller
import label_image

robot = anki_vector.Robot(anki_vector.util.parse_command_args().serial, enable_camera_feed=True)
screen_dimensions = anki_vector.screen.SCREEN_WIDTH, anki_vector.screen.SCREEN_HEIGHT
current_directory = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(current_directory, 'resources', "latest.jpg")

def detect_labels(path):
    print('Detect labels, image = {}'.format(path))
    # file_name = "grace_hopper.jpg"
    model_file = "inception_v3_2016_08_28_frozen.pb"
    label_file = "imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"

    graph = label_image.load_graph(model_file)
    t = label_image.read_tensor_from_image_file(
        path,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = label_image.load_labels(label_file)
        
        return_labels = []
        for i in top_k:
            return_labels.append({ 'label': labels[i], 'prob': results[i] })
            print(labels[i], results[i])

        return return_labels

# def detect_labels(path):
#     print('Detect labels, image = {}'.format(path))
#     # Instantiates a client
#     # [START vision_python_migration_client]
#     client = vision.ImageAnnotatorClient()
#     # [END vision_python_migration_client]

#     # Loads the image into memory
#     with io.open(path, 'rb') as image_file:
#         content = image_file.read()

#     image = types.Image(content=content)

#     # Performs label detection on the image file
#     response = client.label_detection(image=image)
#     labels = response.label_annotations

#     res_list = []
#     for label in labels:
#         if label.score > 0.5:
#             res_list.append(label.description)

#     print('Labels: {}'.format(labels))
#     return ', or '.join(res_list)


# def localize_objects(path):
#     print('Localize objects, image = {}'.format(path))
#     client = vision.ImageAnnotatorClient()

#     with open(path, 'rb') as image_file:
#         content = image_file.read()
#     image = vision.types.Image(content=content)

#     objects = client.object_localization(image=image).localized_object_annotations

#     res_list = []
#     print('Number of objects found: {}'.format(len(objects)))
#     for object_ in objects:
#         print('\n{} (confidence: {})'.format(object_.name, object_.score))
#         print('Normalized bounding polygon vertices: ')
#         res_list.append(object_.name)
#         for vertex in object_.bounding_poly.normalized_vertices:
#             print(' - ({}, {})'.format(vertex.x, vertex.y))

#     return ', and '.join(res_list)


def connect_robot():
    print('Connect to Vector...')
    robot.connect()


def disconnect_robot():
    robot.disconnect()
    print('Vector disconnected')


def stand_by():
    # If necessary, move Vector's Head and Lift to make it easy to see his face
    robot.behavior.set_lift_height(0.0)


def show_camera():
    print('Show camera')
    robot.camera.init_camera_feed()
    robot.vision.enable_display_camera_feed_on_face(True)


def close_camera():
    print('Close camera')
    robot.vision.enable_display_camera_feed_on_face(False)
    robot.camera.close_camera_feed()


def save_image(file_name):
    print('Save image')
    robot.camera.latest_image.save(file_name, 'JPEG')


def show_image(file_name):
    print('Show image = {}'.format(file_name))

    # Load an image
    image = Image.open(file_name)

    # Convert the image to the format used by the Screen
    print("Display image on Vector's face...")
    screen_data = anki_vector.screen.convert_image_to_screen_data(image.resize(screen_dimensions))
    robot.screen.set_screen_with_image_data(screen_data, 20.0, True)


def robot_say(text):
    print('Say {}'.format(text))
    robot.say_text(text)

def robot_driveoff():
    robot_say('I am out of here.')
    robot.behavior.drive_off_charger()

def robot_driveon():
    robot_say('That''s it. I''m headed home.')
    robot.behavior.drive_on_charger()

def robot_drive_straight():
    robot_say('I''m driving straight into oblivion.')
    robot.behavior.drive_straight(distance_mm(200), speed_mmps(100))

def analyze():
    stand_by()
    show_camera()
    robot_say('Preparing to take a photo')

    save_image(image_file)
    show_image(image_file)
    time.sleep(1)

    robot_say('Analyzing')
    labels = detect_labels(image_file)
    robot_say('Here are all the things I see:')
    show_image(image_file)

    for label in labels:
        robot_say("%s, %s percent chance" % (label['label'], math.trunc(label['prob']*100)))

    close_camera()


def main():
    while True:
        connect_robot()
        try:
            analyze()
        except Exception as e:
            print('Analyze Exception: {}', e)

        duration = random.randint(1, 10)
        robot_say('Going back to being normal for %s seconds' % duration)

        disconnect_robot()

        time.sleep(duration)


if __name__ == "__main__":
    main()
