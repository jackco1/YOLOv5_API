# YOLOv5_API

1. Download dependencies for API and YOLOv5 (pip install -r https://raw.githubusercontent.com/kd3n/YOLOv5_API/main/requirements.txt)

2. Download a YOLOv5 model (https://github.com/ultralytics/yolov5/releases). If using a model other than 5x, the relevant variable in detect.py must be changed to reflect it.

3. Run api.py and send a post request containing a JSON body with a base64 encoded image to localhost:5000/image. For example:
{
  "image": "base64data"
}
