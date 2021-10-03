import cv2
import argparse
import base64
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--path",type=str)
args = parser.parse_args()

def img2b64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return(jpg_as_text)

while True:
    img = cv2.imread(args.path)

    op = [{"img":img2b64(img)}]

    resp = json.dumps(str(op))
    response = requests.post('http://127.0.1.1:6969/debrisdet', data=resp)
    result = eval(response.content)
    print(result)