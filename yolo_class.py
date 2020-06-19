# import required packages
import cv2
import argparse
import numpy as np

weights = './yolo_model/tiny-yolo-crack-deep_best.weights'
config = './yolo_model/tiny-yolo-crack-deep.cfg'
name_classes = './yolo_model/obj.names'


class Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class CandidateInformation:
    def __init__(self, class_id, confidence, box):
        self.class_id = class_id
        self.confidence = confidence
        self.box = box


def predict_image(imagePath, scale):
    import time
    # read input image
    image = cv2.imread(imagePath)

    Width = image.shape[1]
    Height = image.shape[0]


    start = time.time()

    blob = cv2.dnn.blobFromImage(image, scale, (Width, Height), (0, 0, 0), True, crop=False)
    net.setInput(blob)


    # function to get the output layer names
    # in the architecture
    def get_output_layers(net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    classes_box = []
    conf_threshold = 0.7
    nms_threshold = 0.7

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                box = Box(x, y, w, h)
                classes_box.append(box)
                boxes.append([x, y, w, h])


    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    find_candidates = []
    coarse_candidates = []
    for i in range(len(class_ids)):
        if class_ids[i] == 0:
            find_candidates.append(CandidateInformation(class_ids[i], confidences[i], classes_box[i]))
        elif class_ids[i] == 1:
            coarse_candidates.append(CandidateInformation(class_ids[i], confidences[i], classes_box[i]))
    # input candidates in the list to divide class name

    if len(classes_box) <= 0:
        return
    # if candidates is empty, don't run next step.

    # # candidates merge algorithm
    # if len(find_candidates) >= 1:
    #     min_x1 = find_candidates[0].box.x1
    #     min_y1 = find_candidates[0].box.y1
    #     max_x2 = find_candidates[0].box.x1 + find_candidates[0].box.x2
    #     max_y2 = find_candidates[0].box.y2 + find_candidates[0].box.y2
    #     # max_width = find_candidates[0].box.x2
    #     # max_height = find_candidates[0].box.y2
    #     for f in find_candidates:
    #         if f.box.x1 < min_x1:
    #             min_x1 = f.box.x1
    #         if f.box.y1 < min_y1:
    #             min_y1 = f.box.y1
    #
    #         if f.box.x2 > max_x2:
    #             max_x2 = f.box.x2
    #         if f.box.y2 > max_y2:
    #             max_y2 = f.box.y2
    #
    #     if min_x1 < 0:
    #         min_x1 = 5
    #     if min_y1 < 0:
    #         min_y1 = 5
    #
    #     # must change confidences[i] -> individual percent
    #     draw_bounding_box(image, 0, confidences[i], round(min_x1), round(min_y1), round(min_x1 + max_x2), round(min_y1 + max_y2))
    #
    # if len(coarse_candidates) >= 1:
    #     min_x1 = coarse_candidates[0].box.x1
    #     min_y1 = coarse_candidates[0].box.y1
    #     max_width = coarse_candidates[0].box.x2
    #     max_height = coarse_candidates[0].box.y2
    #
    #     for c in coarse_candidates:
    #         if c.box.x1 < min_x1:
    #             min_x1 = c.box.x1
    #         if c.box.y1 < min_y1:
    #             min_y1 = c.box.y1
    #
    #         if c.box.x2 > max_width:
    #             max_width = c.box.x2
    #         if c.box.y2 > max_height:
    #             max_height = c.box.y2
    #
    #     if min_x1 < 0:
    #         min_x1 = 5
    #     if min_y1 < 0:
    #         min_y1 = 5
    #     # must change confidences[i] -> individual percent
    #     draw_bounding_box(image, 1, confidences[i], round(min_x1), round(min_y1), round(min_x1 + max_width), round(min_y1 + max_height))


    # class_ids = 0 - fine crack, 1 - coarse crack
    print(class_ids)
    print(confidences)
    print(boxes)

    # go through the detections remaining
    # after nms and draw bounding box



    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))


    print('image predicting time : ', time.time() - start)
    # display output image
    cv2.imshow("object detection", image)

    # wait until any key is pressed
    cv2.waitKey()

    # save output image to disk
    #cv2.imwrite("object-detection.jpg", image)

    # release resources
    #cv2.destroyAllWindows()



# read class names from text file
classes = None
with open(name_classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config)


predict_image('D:\Anaconda3\Scripts\CrackDetection/testData/222.cracks-in-concrete.jpg', scale=0.004)

files = './testData'

import os

f = os.listdir(files)

for a in f:
    predict_image(files + '/' + a, scale=0.019)

