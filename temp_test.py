import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy import argmax

model = load_model(r'C:\Users\Kuro\Downloads\FL\players-matching\players-matching\files\model_6.h5')

# cap = cv2.VideoCapture(r'data\test1.avi')
temp = cv2.imread(r'data\temp.jpg', 0)
ground = cv2.imread(r'data\dst.jpg')

wt, ht = temp.shape[::-1]
img = cv2.imread('data/test_color.jpg')
frame_width = int(img.shape[1])
frame_height = int(img.shape[0])

# if you want to write video
# out = cv2.VideoWriter('match.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1920, 1080))
# out2 = cv2.VideoWriter('plane.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (900, 600))

# if (cap.isOpened() == False):
#     print("Error opening video stream or file")

# Load Yolo
net = cv2.dnn.readNet("files\yolov3.weights", "files\yolov3.cfg")
classes = []
with open("files\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def plane(players, ball):
    coptemp = ground.copy()
    # matrix=np.array([[ 2.56945407e-01,  5.90910632e-01,  1.94094537e+02],
    #                  [-1.33508274e-02,  1.37658562e+00, -8.34967286e+01],
    #                  [-3.41878940e-05,  1.31509536e-03,  1.00000000e+00]])

    matrix = np.array([[1.71561102e-01, -2.46355624e-17, 2.80883317e+02],
                       [-5.56684219e-03, 8.77400601e-01, -4.65732376e+01],
                       [-1.85561406e-05, -6.35007163e-20, 1.00000000e+00]])

    for p in players:
        x = p[0] + int(p[2] / 2)
        y = p[1] + p[3]
        pts3 = np.float32([[x, y]])
        pts3o = cv2.perspectiveTransform(pts3[None, :, :], matrix)
        x1 = int(pts3o[0][0][0])
        y1 = int(pts3o[0][0][1])
        pp = (x1, y1)
        if (p[4] == 0):
            cv2.circle(coptemp, pp, 15, (255, 0, 0), -1)
        elif p[4] == 1:
            cv2.circle(coptemp, pp, 15, (255, 255, 255), -1)
        elif p[4] == 2:
            # print hakm
            # cv2.circle(coptemp,pp, 15, (0,0,255),-1)
            pass
    if len(ball) != 0:
        xb = ball[0] + int(ball[2] / 2)
        yb = ball[1] + int(ball[3] / 2)
        pts3ball = np.float32([[xb, yb]])
        pts3b = cv2.perspectiveTransform(pts3ball[None, :, :], matrix)
        x2 = int(pts3b[0][0][0])
        y2 = int(pts3b[0][0][1])
        pb = (x2, y2)
        cv2.circle(coptemp, pb, 15, (0, 0, 0), -1)
    return coptemp


def get_players(outs, height, width):
    class_ids = []
    confidences = []
    boxes = []
    players = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'person':
                players.append(boxes[i])

    return players


opr = 0
# while (cap.isOpened()):
#     ret, frame = cap.read()
frame = cv2.imread('data/test_color.jpg')

players = []
ball = []

boundaries = [
    ([17, 15, 75], [50, 56, 200]), #red
    ([43, 31, 4], [250, 88, 50]), #blue
    ([187,169,112],[255,255,255]) #white
    ]

copy = frame.copy()
img_color = frame.copy()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

height, width, channels = frame.shape
print(height, width)

blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)
outs = get_players(outs, height, width)
for i in range(len(outs)):
    x, y, w, h = outs[i]
    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, (120,240))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    # cv2.imshow("hsv", hsv)
    # define blue color range
    # light_blue = np.array([136, 87, 111])
    # dark_blue = np.array([180, 255, 255])
    # mask = cv2.inRange(hsv, light_blue, dark_blue)
    # kernal = np.ones((5, 5), "uint8")
    # mask = cv2.dilate(mask, kernal)
    # output = cv2.bitwise_and(roi, roi, mask=mask)
    #
    # cv2.imshow("Color Detected", np.hstack((roi, output)))

    # cv2.imshow("123",roi)
    # cv2.waitKey(0)
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(roi, lower, upper)
        output = cv2.bitwise_and(roi, roi, mask=mask)
        print(output)
        # show the images
        cv2.imshow("images", np.hstack([roi, output]))
        cv2.waitKey(0)

    # some frames are bad so resize function throw an error

    # try:
    #     roi = cv2.resize(roi, (96, 96))
    # except:
    #     continue
    # ym = model.predict(np.reshape(roi, (1, 96, 96, 3)))
    # ym = argmax(ym)
    # players.append([x, y, w, h, ym])
    # cv2.rectangle(copy, (x, y), (x + w, y + h), (70, 150, 255), 2)



# for (lower, upper) in boundaries:
#     lower = np.array(lower, dtype = "uint8")
#     upper = np.array(upper, dtype = "uint8")
#     # find the colors within the specified boundaries and apply
#     # the mask
#     mask = cv2.inRange(img_color, lower, upper)
#     output = cv2.bitwise_and(img_color, img_color, mask = mask)
#     # show the images
#     cv2.imshow("images", np.hstack([img_color, output]))

# cv2.imshow('img', copy)
#     cv2.waitKey(0)
# cv2.imshow('plane', p)

















