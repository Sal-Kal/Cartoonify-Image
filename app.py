import numpy as np
from sklearn.cluster import KMeans
import cv2 as cv
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import base64

app = Flask(__name__)
CORS(app)

@app.route("/cartoonify", methods = ['POST'])
def cartoonify():
    try:
        response = {}
        image = request.files["image"].read()
        nparr = np.frombuffer(image, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        line_size = 7
        blur_value = 7

        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray_blur = cv.medianBlur(gray_img, blur_value)
        edges = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, line_size, blur_value)

        k = 7
        data = img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        img_reduced = kmeans.cluster_centers_[kmeans.labels_]
        img_reduced = img_reduced.reshape(img.shape)
        img_reduced = img_reduced.astype(np.uint8)

        blurred = cv.bilateralFilter(img_reduced, d=7, sigmaColor=200,sigmaSpace=200)
        cartoon = cv.bitwise_and(blurred, blurred, mask=edges)

        cartoon_ = cv.cvtColor(cartoon, cv.COLOR_RGB2BGR)
        _, buffer = cv.imencode('.png', cartoon_)
        encoded = base64.b64encode(buffer).decode('utf-8')

        response["status"] = "succes"
        response["response"] = encoded

        return jsonify(response)

    except Exception as e:
        response = {}
        response["status"] = "error"
        response["response"] = str(e)
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=8000, host="0.0.0.0")