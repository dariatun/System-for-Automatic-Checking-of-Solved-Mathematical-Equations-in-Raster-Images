import cv2
import subprocess
from tkinter import *
from PIL import ImageTk, Image
from handwritten_recogniser import handwritten_recogniser
from utils import rotate_img, change_size_img
from check_correctness import correctness
from scan import scan
from another_scan import another_scan

# Windows dependencies
# - Python 2.7.6: http://www.python.org/download/
# - OpenCV: http://opencv.org/
# - Numpy -- get numpy from here because the official builds don't support x64:
#   http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

# Mac Dependencies
# - brew install python
# - pip install numpy
# - brew tap homebrew/science
# - brew install opencv
"""
root = Tk()
canvas = Canvas(root, width=300, height=300)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("capture.jpg"))
canvas.create_image(20, 20, anchor=NW, image=img)
"""


class App(Frame):
    def __init__(self, master=None):
        self.leave_loop = False
        Frame.__init__(self, master)
        self.master = master
        self.widgets()
        """
        self.canvas = Canvas(root, width=300, height=300)
        self.img = ImageTk.PhotoImage(Image.open("capture.jpg"))
        self.img_area = self.canvas.create_image(20, 20, anchor=NW, image=self.img)
        self.canvas.pack()
        self.but1 = Button(root, text="press me", command=lambda: self.end_loop())
        self.but1.place(x=10, y=500)
        """
        #root.after(2000, self.while_loop())

    def widgets(self):
        self.master.title("App")
        self.pack(fill=BOTH, expand=1)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Exit", command=self.client_exit)

        menu.add_cascade(label="File", menu=file)

        edit = Menu(menu)
        edit.add_command(label="Start while loop", command=self.while_loop)
        edit.add_command(label="Leave loop", command=self.end_loop)
        #edit.add_command(label="Show image", command=self.show_img)
        #edit.add_command(label="Show text", command=self.show_text)
        # edit.add_command(label="Undo")
        menu.add_cascade(label="Edit", menu=edit)
        self.cap = cv2.VideoCapture(0)

        #self.load_images()

    def load_images(self):
        self.load_image("capture.jpg", 0, 0)
        self.load_image("edged.jpg", 600, 0)
        self.load_image("countour.jpg", 0, 400)
        self.load_image("out/capture_rec.jpg", 600, 400)

    def load_image(self, filename, x, y):
        load = Image.open(filename)
        load = change_size_img(load, 600)
        render = ImageTk.PhotoImage(load)

        img = Label(self, image=render)
        img.image = render
        img.place(x=x, y=y)

    def end_loop(self):
        self.leave_loop = True

    def client_exit(self):
        exit()

    def while_loop(self):
        #while not self.leave_loop:

            ret, frame = self.cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            # cv2.imshow('frame', rgb)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('capture.jpg', frame)
            # break

            rotate_img('capture.jpg', 180)
            another_scan('capture.jpg', 'capture.jpg')
            #if not scan('capture.jpg'):
            #    self.load_images()
            #    self.while_loop()
            #    return
                #continue
            rc = subprocess.call(". script.sh", shell=True)

            digit_predictions, digit_xy_coords, equations_predictions, equations_xy_coords = handwritten_recogniser()
            correctness(digit_predictions, digit_xy_coords, equations_predictions, equations_xy_coords)

            self.load_images()
            print('End')
            #self.img = PhotoImage(file="capture.jpg")
            #self.canvas.itemconfig(self.img_area, image=self.img)



root = Tk()
root.geometry("1600x1200")
app = App(root)

root.mainloop()
#cap.release()
cv2.destroyAllWindows()
