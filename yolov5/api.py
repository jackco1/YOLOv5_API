from detect import inference

from flask import Flask, request, render_template

import base64
import json

app = Flask(__name__)

@app.route('/image', methods=['POST'])
def detect():
    data = request.get_json() # receive POST data
    base64decode = base64.b64decode(data['image']) # decode base64 data
    return inference(base64decode) # send to inference script

if __name__ == '__main__':
    app.run(debug=True)
