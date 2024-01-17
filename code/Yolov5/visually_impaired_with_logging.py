import json
import os.path

import cv2
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_sock import Sock
import torch
from matplotlib import pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadImages2
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device

from flask_classful import FlaskView
from PIL import Image
import numpy as np

app = Flask(__name__)
api = Api(app)
sock = Sock(app)
import pusher


# defines which condition is logged
class EXP:
    EXPS = "EXP-S"
    EXPC = "EXP-C"
    EXPM = "EXP-M"
    NONE = "None"


class STATUS:
    START = "START"
    STOP = "STOP"
    NONE = "NONE"
    CAPTURE_START_POSITION = "CAPTURE_START_POSITION"


cache = {
    'distance': 10,
    'lastDistance': -1,
    'status': STATUS.NONE,
    'exp': EXP.NONE,
    'hand': 'left',
    'subject': 'X',
    'setStartPositon': False
}

images_folder = "data/"
logging_folder = "logging/"

coco_classes = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

coco_classes_german = {
    "person": "person",
    "bicycle": "fahrrad",
    "car": "auto",
    "motorcycle": "motorrad",
    "airplane": "flugzeug",
    "bus": "bus",
    "train": "zug",
    "truck": "lastwagen",
    "boat": "boot",
    "traffic light": "ampel",
    "fire hydrant": "hydrant",
    "stop sign": "stopschild",
    "parking meter": "parkuhr",
    "bench": "bank",
    "bird": "vogel",
    "cat": "katze",
    "dog": "hund",
    "horse": "pferd",
    "sheep": "schaf",
    "cow": "kuh",
    "elephant": "elefant",
    "bear": "bär",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "backpack": "rucksack",
    "umbrella": "regenschirm",
    "handbag": "handtasche",
    "tie": "krawatte",
    "suitcase": "koffer",
    "frisbee": "frisbee",
    "skis": "Ski",
    "snowboard": "snowboard",
    "sports ball": "ball",
    "kite": "drachen",
    "baseball bat": "baseball handschuh",
    "baseball glove": "baseball schläger",
    "skateboard": "skateboard",
    "surfboard": "surfbrett",
    "tennis racket": "tennisschläger",
    "bottle": "flasche",
    "wine glass": "weinglas",
    "cup": "tasse",
    "fork": "gabel",
    "knife": "messer",
    "spoon": "löffel",
    "bowl": "schüssel",
    "banana": "banane",
    "apple": "apfel",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "brokkoli",
    "carrot": "karotte",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "kuchen",
    "chair": "stuhl",
    "couch": "sofa",
    "potted plant": "blume",
    "bed": "bett",
    "dining table": "tisch",
    "toilet": "toilette",
    "tv": "tv",
    "laptop": "laptop",
    "mouse": "maus",
    "remote": "fernbedienung",
    "keyboard": "tastatur",
    "cell phone": "telefon",
    "microwave": "mikrowelle",
    "oven": "ofen",
    "toaster": "toaster",
    "sink": "waschbecken",
    "refrigerator": "kühlschrank",
    "book": "buch",
    "clock": "uhr",
    "vase": "vase",
    "scissors": "schere",
    "teddy bear": "teddy bär",
    "hair drier": "fön",
    "toothbrush": "zahnbürste"
}

# load model from given weights
device = select_device("")
# load FP32 model
model = attempt_load("yolov5s.pt", map_location=device)
stride = int(model.stride.max())
# get class names
names = model.module.names if hasattr(model, 'module') else model.names
stride = int(model.stride.max())  # model stride
half = device.type != 'cpu'
if half:
    model.half()  # to FP16


class Detector:
    def __init__(self):
        print("init detector")

    def detect(self, image, scale_percentage=20):
        # print("detecting image")
        # print(image.shape)
        # temp_image = cv2.imread('test.jpg')
        ret_dict = []
        image_path = 'data/ablation_study/1m/20221115_102519_HoloLens.jpg'
        image = cv2.imread(image_path)
        scale_percent = 20  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        print(img.shape)
        dataset = LoadImages2([image], stride=stride)

        # print(names)

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
            # pred.print()
            # exit(0)
            # print(pred)
            pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45, agnostic=False, max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                # print(im0.shape)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = torch.tensor(xyxy).view(-1, 4)
                        res = np.copy(xyxy2xywh(xyxy)[0])  # boxes
                        print(conf)
                        print(cls)
                        # put a blue dot at (10, 20)
                        # cv2.circle(temp_image, (int(res[0]), int(res[1])), 5, (0, 0, 255), -1)
                        # plt.scatter(res[0], res[1])
                        ret_dict.append(
                            {
                                "xCenter": float(res[0]) * 100 / scale_percentage,
                                "yCenter": float(res[1]) * 100 / scale_percentage,
                                "width": float(res[2]) * 100 / scale_percentage,
                                "height": float(res[3]) * 100 / scale_percentage,
                                "classId": names[int(cls)],
                                "className": coco_classes_german[coco_classes[int(cls)]]
                                # "confidence": conf
                            }
                        )

        # cv2.imwrite("temp.jpg", temp_image)

        return ret_dict


detector = Detector()

# show the user profile for that user
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


@app.route('/post/objectcoords', methods=['POST'])
def post_object_coords():
    # start.record()
    # print("post received")
    filestr = request.files['image'].read()
    # convert string data to numpy array
    npimg = np.frombuffer(filestr, np.uint8)

    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    # print(img.shape)
    if img.shape[-1]:
        img = img[..., :3]

    # cv2.imwrite('test.jpg', img)
    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # print(resized.shape)
    ret = detector.detect(img, scale_percent)

    print(ret)
    # print(jsonify(ret))
    return jsonify(ret)

@app.route('/post/eval', methods=['POST'])
def eval():
    # start.record()
    # print("post received")


    # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # print(resized.shape)
    ret = detector.detect([], 20)

    print(ret)
    # print(jsonify(ret))
    return jsonify(ret)

# pusher for publishing to apple watch
pusher_client = pusher.Pusher(
    app_id='1457669',
    key='49671136b3fff637f7d9',
    secret='9b27babb4695f425705f',
    cluster='eu',
    ssl=True
)

@app.route('/post/distance', methods=['POST'])
def post_distance():
    # cache['lastDistance'] = cache['distance']
    cache['distance'] = float(request.form.get('distance').replace(",", "."))
    print("distance")
    print(cache['distance'])
    print("\n")
    intervall = 9.0
    if cache['distance'] < 0.5:
        intervall = 0.4
    elif cache['distance'] < 0.8:
        intervall = 0.7
    elif cache['distance'] < 1.5:
        intervall = 1.4
    elif cache['distance'] < 2.0:
        intervall = 1.9
    else:
        intervall = 9.0

    if intervall != cache['lastDistance']:
        cache['lastDistance'] = intervall
        print("SEND DISTANCE " + str(intervall))
        pusher_client.trigger('my-channel', 'my-event', {
            'name': "hololens",
            'distanceInMeter': intervall
        })
    return 'Success'

@app.route('/post/logging', methods=['POST'])
def post_logging():
    # print(request.values)
    file = request.files['file']
    # print(file.filename)
    if file.filename != '':
        file_path = os.path.join(logging_folder, file.filename.split('/')[-1])
        file.save(file_path)
    return jsonify({
        '200': 'Success'
    })

@app.route('/post/status', methods=['POST'])
def post_status():
    cache['status'] = request.values['status']
    cache['exp'] = request.values['exp']
    cache['subject'] = request.values['subject']
    cache['hand'] = request.values['hand']
    return str(cache['status']) + ' ' + str(cache['exp'])

@sock.route('/socket/status')
def sendStatus(sock):
    while True:
        if cache['status'] == STATUS.START or cache['status'] == STATUS.STOP or cache['status'] == STATUS.CAPTURE_START_POSITION:
            # sock.send("EXP-S")
            print("SEND STATUS")
            sock.send(json.dumps(
                {
                    'name': cache['exp'],
                    'status': cache['status'],
                    'subject': cache['subject'],
                    'hand': cache['hand']
                }
            ))
            cache['status'] = STATUS.NONE

if __name__ == '__main__':
    app.run(debug=True) # , host="192.168.1.112")
    #app.run(debug=True, host=" 10.20.55.106")



# 192.168.178.34
