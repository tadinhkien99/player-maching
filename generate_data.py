import cv2
import numpy as np
from PIL import Image





# img = cv2.imread('data/test_color.jpg')
# frame_width = int(img.shape[1])
# frame_height = int(img.shape[0])


net = cv2.dnn.readNet("files\yolov3.weights", "files\yolov3.cfg")
classes = []
with open("files\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


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

cap = cv2.VideoCapture('data/videos/video_3.mp4')
count=1800
frame_count = 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame_count+=1
    # frame = cv2.imread('data/test_color.jpg')
        if frame_count%20==0:
            players = []
            copy = frame.copy()
            img_color = frame.copy()

            height, width, channels = frame.shape
            # print(height, width)

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)
            outs = get_players(outs, height, width)
            for i in range(len(outs)):
                x, y, w, h = outs[i]

                roi = frame[y:y + h, x:x + w]
                # roi = cv2.resize(roi, (60,120))
                try:
                    roi = cv2.resize(roi, (96, 96))
                except:
                    continue
                cv2.imwrite("train-classification_model/data/" + str(count) + ".jpg", np.array(roi))
                count+=1
                print(count)
                # cv2.waitKey(0)
                # im.save("train-classification_model/data/" + str(i) + ".jpg")
    else:
        break


















