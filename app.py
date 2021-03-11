# encoding:utf-8
# !/usr/bin/env python
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
import time
import os
import base64
import cv2 as cv
import os
import face_recognition
from lbp import LBP
import datetime
import random
from datetime import timedelta

def create_uuid():  # 生成唯一的图片的名称字符串，防止图片显示时的重名问题
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
    randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum


app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
lbp = LBP(20,(50,50),5, 'static/images')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def upload_test():
    return render_template('index.html')


# 上传文件
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def api_upload():
    f = request.files['photo']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        ext = fname.rsplit('.', 1)[1]
        filename = create_uuid() + '.' + ext
        new_filename =  'static/' + filename
        new_face = 'static/face_' + filename
        f.save(new_filename)
        labels = []
        image = cv.imread(new_filename)
        faces = face_recognition.face_locations(image)
        for i, (top, right, bottom, left) in enumerate(faces):
            cv.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)
            temp = image[top:bottom, left:right]
            label = lbp.predict(temp)
            labels.append(label[14:])
            cv.putText(image, label[14:], (left, top), cv.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1, 2)
        cv.imwrite(new_face, image)
        return render_template('face.html', new_face=new_face, file_name=new_filename, labels = labels)
    else:
        return jsonify({"error": 1001, "msg": f"上传失败"})



if __name__ == '__main__':
    app.run(debug=True)