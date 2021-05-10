import numpy as np
import tensorflow as tf
import cv2
from model import get_segmentation_model, get_detection_model
import base64
from PIL import Image
import io
from flask import jsonify


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

s_model = get_segmentation_model()
d_model = get_detection_model()

def image_upload_handler(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    img1 = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_UNCHANGED)
    frame = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    inp = tf.keras.preprocessing.image.img_to_array(frame)
    res = s_model.predict(np.expand_dims(inp, axis=0))
    res = res * 255
    redImg = np.zeros(inp.shape, inp.astype('uint8').dtype)
    redImg[:,:] = (0, 0, 255)
    redMask = cv2.bitwise_and(redImg, redImg, mask=np.squeeze(res).astype('uint8'))
    mask_t = cv2.addWeighted(redMask, 0.5, inp.astype('uint8'), 0.5, 0, inp.astype('uint8'))
    mask_t = cv2.cvtColor(mask_t, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(mask_t.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame.astype("uint8"))
    rawBytes2 = io.BytesIO()
    img2.save(rawBytes2, "JPEG")
    rawBytes2.seek(0)
    img2_base64 = base64.b64encode(rawBytes2.read())
    frame = cv2.resize(img1, (256, 256))
    inp = tf.keras.preprocessing.image.img_to_array(frame) / 255.0
    res = d_model.predict(np.expand_dims(inp, axis=0))
    return jsonify({'status':img_base64.decode("utf-8"), 'inp': img2_base64.decode("utf-8"), 'labels': res.tolist()})