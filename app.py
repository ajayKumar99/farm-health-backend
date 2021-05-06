import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, render_template, Response, request, make_response
from videoHandler import get_prediction_video, get_prediction_label, get_live_video, get_live_label
from imageHandler import image_upload_handler
import base64
import urllib
from werkzeug.utils import secure_filename

if not os.path.exists('uploads'):
    os.makedirs('uploads')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/video")
def video():
    return Response(get_prediction_video("uploads/test.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/live")
def live():
    return Response(get_live_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_label")
def video_label():
    return Response(get_prediction_label("uploads/test.mp4"), mimetype="text/event-stream", headers={"Access-Control-Allow-Origin": "*"})

@app.route("/live_label")
def live_label():
    return Response(get_live_label(), mimetype="text/event-stream", headers={"Access-Control-Allow-Origin": "*"})

@app.route("/image_upload", methods=['POST'])
def image_upload():
    resp = make_response(image_upload_handler(request.files['image'].read()))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/video_upload", methods=['POST'])
def video_upload():
    file = request.files['video']
    content = file.stream
    location = content.name
    if file and allowed_file(file.filename):
        file.filename = "test.mp4"
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    location = base64.b64encode(location.encode('ascii'))
    location = urllib.parse.quote_plus(location.decode('utf-8'))
    resp = make_response({"local_temp": location})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(debug=True)