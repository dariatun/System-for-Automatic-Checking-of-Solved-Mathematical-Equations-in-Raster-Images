import cv2
import numpy as np
import time
import os
import subprocess
import json
from PIL import Image
from another_scan import another_scan
from utils import rotate_img_opencv
from handwritten_recogniser import recognise_object

colors = np.random.uniform(0, 255, size=(2, 3))

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
    cv2.imwrite('capture.jpg', frame)

    height, width, channels = frame.shape

    # Detecting objects
    #blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    #net.setInput(blob)
    #outs = net.forward(output_layers)
    rc = subprocess.call(". script.sh", shell=True)
    path_to_image_folder = '/Users/dariatunina/mach-lerinig/mLStuff/'
    path_to_json_file = '/Users/dariatunina/mach-lerinig/mLStuff/result.json'
    with open(path_to_json_file) as json_file:
        data = json.load(json_file)
        for file in data:
            filename = file['filename'].split('/')[-1]
            outs = file['objects']
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    predictions = []
    for out in outs:
        #for detection in out:
            # scores = detection[5:]
            class_id = out['class_id']#np.argmax(scores)
            confidence = out['confidence']#scores[class_id]
            if confidence > 0.2:
                # Object detected
                coordinates = out['relative_coordinates']
                center_x = int(coordinates['center_x'] * width)
                center_y = int(coordinates['center_y'] * height)
                w = int(coordinates['width'] * width)
                h = int(coordinates['height'] * height)

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

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        # label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        prediction = predictions[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
        cv2.putText(frame, str(prediction), (x, y + 30), font, 3, (255, 255, 255), 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(fps), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("App uno", frame)
    filename = str(count) + '.jpg'
    count += 1
    cv2.imwrite("out/" + filename, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

