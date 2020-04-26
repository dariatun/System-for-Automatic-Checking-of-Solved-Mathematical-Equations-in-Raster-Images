import cv2
import numpy as np
import time
import os
from PIL import Image
from another_scan import another_scan
from utils import rotate_img_opencv
from handwritten_recogniser import recognise_object

# Load Yolo

net = cv2.dnn.readNet("/Users/dariatunina/mach-lerinig/darknet/yolov3-full1_best.weights",
                      "/Users/dariatunina/mach-lerinig/darknet/cfg/yolov3-full1.cfg")
classes = []
with open("/Users/dariatunina/mach-lerinig/darknet/data/obj-full1.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading camera
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
count = 0
while True:
    _, frame = cap.read()
    frame_id += 1
    frame = rotate_img_opencv(frame, 180)
    frame = another_scan('capture.jpg', 'capture.jpg', frame)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    predictions = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                filename = "{}.jpg".format(os.getpid())
                cv2.imwrite(filename, frame)
                prediction = recognise_object(np.array(Image.open(filename)), x, y, w, h, class_id)
                os.remove(filename)
                # prediction = 'hey'
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                predictions.append(prediction)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            prediction = predictions[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, str(prediction), (x, y + 30), font, 3, (255, 255, 255), 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(fps), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Another app", frame)
    filename = str(count) + '.jpg'
    count += 1
    cv2.imwrite("out/" + filename, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

