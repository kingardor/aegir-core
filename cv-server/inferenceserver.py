import os
import time
from PIL import Image
from io import BytesIO
import base64
import json
import cv2
import numpy as np
import argparse
import importlib
from loguru import logger
import torch
from torch2trt import TRTModule

from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
flaskserver = Flask(__name__)
cors = CORS(flaskserver)

global utility, debris

class Utility:
    """ Class to hold utility functions """
    def __init__(self):
        pass

    def base64_to_image(self, base64_string):
        ''' Take in base64 string and return PIL image. '''

        img =  np.asarray(Image.open(BytesIO(base64.b64decode(base64_string))))
        if len(img.shape) > 2 and img.shape[2] == 4:
            #convert the image from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

class DebrisInference:
    """ Class to hold inference functions """

    def __init__(self):
        self.exp = None
        self.model = None

        self.nms = 0.3
        self.conf = 0.3
        self.imgsize = 640

        self.CLASSES = ("bottle", "garbage")

        self.num_classes = len(self.CLASSES)
        self.preproc = ValTransform(legacy=False)

        # Load the model
        self.load_model()

    def load_model(self):
        exp_file = 'yolox_s_debris.py'
        trt_file = 'model_debris.pth'
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        self.exp = current_exp.Exp()
        self.model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
        self.model.cuda()
        # self.model.half()
        self.model.eval()
        self.model.head.decode_in_inference = False
        self.decoder = self.model.head.decode_outputs
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(trt_file))
        x = torch.ones(1, 3, self.imgsize, self.imgsize).cuda()
        self.model(x)
        self.model = self.model_trt
        logger.info('Model initialised.')

    def inference(self, img):
        img_info = {"id": 0}

        # Read image

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.imgsize / img.shape[0], self.imgsize / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, (self.imgsize, self.imgsize))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()
        # img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.conf,
                self.nms, class_agnostic=True
            )
        logger.info("Inference time: {:.4f}s".format(time.time() - t0))

        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

@flaskserver.route(rule='/debrisdet', methods=['POST'])
@cross_origin()
def debris_detector():
    """ Method to run inference to detect debris"""

    return_list = list()

    image_requests = request
    images = eval(image_requests.data.decode('utf-8'))
    images = eval(images) if type(images)==str else images

    # Get image from base64
    img = utility.base64_to_image(base64_string=images[0]['img'])
    outputs, img_info = debris.inference(img)
    outputs = outputs[0]
    if not outputs == None:
        outputs = outputs.cpu().tolist()
        ratio = img_info["ratio"]
        # result_frame = debris.visual(outputs[0], img_info)

        for out in outputs:
            result = dict()
            result.update({'x0':out[0] / ratio})
            result.update({'y0':out[1] / ratio})
            result.update({'x1':out[2] / ratio})
            result.update({'y1':out[3] / ratio})
            result.update({'score':out[4] * out[5]})
            result.update({'class':int(out[6])})
            return_list.append(result)

    return json.dumps(return_list)

utility = Utility()
debris = DebrisInference()

if __name__ == '__main__':
    flaskserver.run(host='127.0.0.1',
                    port=5000,
                    debug=True)
