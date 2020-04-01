import numpy as np
import cv2
import os
from flask import Flask, json, request
import base64

app = Flask(__name__)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)    # return the resized image
    return resized


def base_64_to_img(base_64_string):
    """Return an base64 image string into cv2 object"""
    # convert image into np array
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(base_64_string.split(";base64,").pop()), np.uint8),
        cv2.IMREAD_COLOR)


ear_cascade = cv2.CascadeClassifier('cascade_file/cascade.xml')

@app.route('/api/getCordinates', methods=['POST'])
def getCordinates():
    base_img = request.get_json()
    img = base_64_to_img(base_img["base64img"])
    # img = cv2.imread("image_2.jpg")
    img = image_resize(img, width=720)
    Lears= ear_cascade.detectMultiScale(img, 1.3, 5)
    x,y,w,h = Lears[0]
    return {"x":int(x+w/4), "y":int((y+h) - 25)}

@app.route("/api", methods=["GET"])
def home():
    return "Hello This is working"


if __name__ == "__main__":
    app.run(debug=True)