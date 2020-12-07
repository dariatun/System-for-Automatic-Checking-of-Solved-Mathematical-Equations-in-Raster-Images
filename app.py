import time
from tkinter import *
import datetime
import cv2
import numpy as np
import os
from utils import change_size_img
from another_scan import another_scan
from utils import rotate_img_opencv
from handwritten_recogniser import recognise_object
from PIL import ImageTk, Image
from check_correctness import correctness, correctness_one
from dataclasses import dataclass





TEACHER_MODE = True

CORRECT_ANSWER = 'Y'
INCORRECT_ANSWER = 'N'
UNDECIDED_ANSWER = 'U'

@dataclass
class Object:
    box: []
    class_id: int
    confidence: float
    prediction: []
    corr_answer: UNDECIDED_ANSWER

class App(Frame):

    def __init__(self, master=None, **kw):
        super().__init__(**kw)
        self.root = master
        self.widgets()
        self.mode = TEACHER_MODE
        master.configure(background='black')
        self.cap = cv2.VideoCapture(0)

        lab = Label(master)
        lab.pack()
        self.net = cv2.dnn.readNet("/Users/dariatunina/mach-lerinig/darknet/yolov3-full1_best.weights",
                                   "/Users/dariatunina/mach-lerinig/darknet/cfg/yolov3-full1.cfg")
        self.classes = []
        with open("/Users/dariatunina/mach-lerinig/darknet/data/obj-full1.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes) + 1, 3))

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.starting_time = time.time()
        self.frame_id = 0

        self.leave_button = Button(master, text="End loop", command=lambda: self.end_loop())
        self.leave_button.place(x=10, y=550)

        self.start_button = Button(master, text="Start loop", command=lambda: self.start_loop())
        self.start_button.place(x=10, y=500)

        self.leave_loop = False
        self.var = StringVar()
        self.fpd_label = Label(master, textvariable=self.var, relief=RAISED)
        self.fpd_label.place(x=0, y=0)

        self.objects = []

        self.rows = 0
        self.columns = 0

        self.pred_matrix = []
        # self.fpd_label.pack()
        # self.clock()
        _, frame = self.cap.read()
        cv2.imwrite("capture.jpg", frame)
        load = Image.open("capture.jpg")
        load = load.resize((600, 300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(master)
        img.image = render
        img.place(x=0, y=100)

    def widgets(self):
        self.root.title("App")
        # self.pack(fill=BOTH, expand=1)

        menu = Menu(self.root)
        self.root.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Teacher mode", command=self.turn_on_teacher_mode)
        file.add_command(label="Debug mode", command=self.turn_on_debug_mode)

        file.add_command(label="Exit", command=self.client_exit)

        menu.add_cascade(label="File", menu=file)

    def turn_on_teacher_mode(self):
        self.mode = TEACHER_MODE
        self.root.configure(background='black')

    def turn_on_debug_mode(self):
        self.mode = not TEACHER_MODE
        self.root.configure(background='white')

    def client_exit(self):
        exit()

    def end_loop(self):
        self.leave_loop = True

    def start_loop(self):
        self.leave_loop = False
        self.clock()

    def append_box(self, prev_b, b):
        new_b = [0] * 4
        new_b[0] = prev_b[0]
        new_b[1] = prev_b[1] if prev_b[1] < b[1] else b[1]
        new_b[2] = prev_b[2] + b[2]
        new_b[3] = prev_b[3] if prev_b[3] < b[3] else b[3]
        return new_b

    def get_boxes(self, objects):
        boxes = []
        for obj in objects:
            boxes.append(obj.box)
        return boxes

    def get_conf(self, objects):
        boxes = []
        for obj in objects:
            boxes.append(obj.confidence)
        return boxes

    def clock(self):

        _, frame = self.cap.read()
        self.frame_id += 1
        frame = rotate_img_opencv(frame, 180)
        # frame = cv2.cvtColor(cv2.imread("countour.jpeg"), cv2.COLOR_BGR2RGB)
        frame = another_scan('capture.jpg', 'capture.jpg', frame)
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        objects = []
        new_pred_matrix = []
        prev_coord = [0, 0]
        prev_row_indx = 0
        prev_col_indx = 0
        prev_class_id = -1
        first = True
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    box = [x, y, w, h]
                    objects.append(Object(box, class_id, float(confidence), '', UNDECIDED_ANSWER))

        objects = sorted(objects, key=lambda k: [k.box[1], k.box[0]])
        prev_obj = None
        for obj in objects:
            class_id = obj.class_id
            box = obj.box
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            filename = "{}.jpg".format(os.getpid())
            cv2.imwrite(filename, frame)
            prediction = recognise_object(np.array(Image.open(filename)), x, y, w, h, class_id)
            os.remove(filename)
            if (prediction is None or prediction == ""):
                objects.remove(obj)
                continue
            else:
                # answers.append(UNDECIDED_ANSWER)
                box = [x, y, w, h]
                if class_id == 0:
                    obj.prediction = prediction
                    if first:
                        new_pred_matrix.append([prediction, ''])
                        prev_col_indx = 1
                        first = False
                    else:
                        if abs(prev_coord[0] - box[0]) > abs(prev_coord[1] - box[1]):
                            new_pred_matrix.append([prediction, ''])
                            prev_row_indx += 1
                            prev_col_indx = 1
                        else:
                            new_pred_matrix[prev_row_indx].append(prediction)
                            new_pred_matrix[prev_row_indx].append('')
                            prev_col_indx += 2
                    prev_coord = [box[0] + box[2], box[1]]
                    prev_class_id = 0
                    prev_obj = obj

                else:
                    #if self.mode is TEACHER_MODE:
                        if prev_row_indx == 0 or prev_col_indx == 0 or prev_class_id == -1:
                            objects.remove(obj)
                            continue
                        if 0 < prev_coord[0] - box[0] < 50 and abs(prev_coord[1] - box[1]) < 10:
                            if prev_class_id == 0:
                                obj.prediction = prediction

                                new_pred_matrix[prev_row_indx][prev_col_indx] = prediction
                                answer = correctness_one(new_pred_matrix[prev_row_indx][prev_col_indx - 1], prediction)
                                obj.corr_answer = (CORRECT_ANSWER if answer else INCORRECT_ANSWER)
                                prev_coord = [box[0] + box[2], box[1]]
                                prev_obj = obj
                                prev_class_id = 1
                            else:
                                prev_obj.box = self.append_box(obj.box, box)
                                prev_obj.prediction += prediction

                                new_pred_matrix[prev_row_indx][prev_col_indx] = prev_obj.prediction
                                answer = correctness_one(new_pred_matrix[prev_row_indx][prev_col_indx - 1], prev_obj.prediction)
                                prev_obj.corr_answer = (CORRECT_ANSWER if answer else INCORRECT_ANSWER)
                                objects.remove(obj)
                        else:
                            objects.remove(obj)
                    #else:
                    #    obj.prediction = prediction

        indexes = cv2.dnn.NMSBoxes(self.get_boxes(objects), self.get_conf(objects), 0.4, 0.3)

        self.objects = objects

        for i in range(len(objects)):
            if i in indexes:
                x, y, w, h = objects[i].box
                if self.mode is not TEACHER_MODE:
                    color = self.colors[objects[i].class_id]
                    prediction = objects[i].prediction
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame, str(prediction), (x, y + 30), self.font, 3, (255, 255, 255), 3)

                if objects[i].class_id == 1:
                    color = self.colors[2]
                    cv2.rectangle(frame, (x + w, y), (x + w * 2, y + h), color, 2)
                    cv2.rectangle(frame, (x + w, y), (x + w * 2, y + 30), color, -1)
                    cv2.putText(frame, objects[i].corr_answer, (x + w, y + 30), self.font, 3, (255, 255, 255), 3)

        elapsed_time = time.time() - self.starting_time
        fps = self.frame_id / elapsed_time
        cv2.imwrite("capture.jpg", frame)

        load = Image.open("capture.jpg")
        load = load.resize((600, 300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        img = Label(image=render)
        img.image = render
        img.place(x=0, y=100)

        if self.mode is not TEACHER_MODE:
            var = StringVar()
            label = Label(root, textvariable=var, relief=RAISED)
            self.var.set("FPS: " + str(fps))

        if not self.leave_loop:
            root.after(10000, self.clock)  # run itself again after 1000 ms


# run first time

root = Tk()
root.geometry("800x600")
app = App(root)

root.mainloop()
# cap.release()
cv2.destroyAllWindows()
