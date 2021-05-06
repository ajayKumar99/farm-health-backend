import tensorflow as tf
import numpy as np
import cv2
from model import get_segmentation_model, get_detection_model
import json
import time


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

s_model = get_segmentation_model()
d_model = get_detection_model()

labels = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']
# cap_live = ""

def get_prediction_video(filename):
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        inp = tf.keras.preprocessing.image.img_to_array(frame)
        res = s_model.predict(np.expand_dims(inp, axis=0))
        res = res * 255
        redImg = np.zeros(inp.shape, inp.astype('uint8').dtype)
        redImg[:,:] = (0, 0, 255)
        redMask = cv2.bitwise_and(redImg, redImg, mask=np.squeeze(res).astype('uint8'))
        mask_t = cv2.addWeighted(redMask, 0.5, inp.astype('uint8'), 0.5, 0, inp.astype('uint8'))
        ret, buffer = cv2.imencode('.jpg', mask_t)
        out = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n'b'Test: test \r\n')

def get_live_video():
    # global cap_live
    cap_live = cv2.VideoCapture(0)
    while (cap_live.isOpened()):
        ret, frame = cap_live.read()
        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        inp = tf.keras.preprocessing.image.img_to_array(frame)
        res = s_model.predict(np.expand_dims(inp, axis=0))
        res = res * 255
        redImg = np.zeros(inp.shape, inp.astype('uint8').dtype)
        redImg[:,:] = (0, 0, 255)
        redMask = cv2.bitwise_and(redImg, redImg, mask=np.squeeze(res).astype('uint8'))
        mask_t = cv2.addWeighted(redMask, 0.5, inp.astype('uint8'), 0.5, 0, inp.astype('uint8'))
        ret, buffer = cv2.imencode('.jpg', mask_t)
        out = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n'b'Test: test \r\n')

def get_prediction_label(filename):
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (256, 256))
        inp = tf.keras.preprocessing.image.img_to_array(frame) / 255.0
        res = d_model.predict(np.expand_dims(inp, axis=0))
        label = labels[np.argmax(res)]
        time.sleep(.1)    
        yield "data: %s \n\n" % (json.dumps({"pred": res.tolist()}))

def get_live_label():
    # global cap_live
    cap_live = cv2.VideoCapture(0)
    while (cap_live.isOpened()):
        ret, frame = cap_live.read()
        frame = cv2.resize(frame, (256, 256))
        inp = tf.keras.preprocessing.image.img_to_array(frame) / 255.0
        res = d_model.predict(np.expand_dims(inp, axis=0))
        label = labels[np.argmax(res)]
        time.sleep(.1)    
        yield "data: %s \n\n" % (json.dumps({"pred": res.tolist()}))