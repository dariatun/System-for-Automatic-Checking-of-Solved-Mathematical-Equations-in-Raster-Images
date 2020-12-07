from tkinter import *
import cv2
from PIL import ImageTk, Image


class App(Frame):

    def __init__(self, master=None, **kw):
        super().__init__(**kw)
        self.root = master
        self.root.configure(background='pink')
        self.root.title("App")

        img_path = '/Users/dariatunina/mach-lerinig/mLStuff/images/emotions/happy.jpeg'
        load = Image.open(img_path)
        load = load.resize((600, 300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self.root, image=render)
        img.place(x=0, y=100)
        #img = Image.open(img_path)
        #img = resize_image(img, 600)

        #panel = Label(self.root, image=ImageTk.PhotoImage(img))
        #panel.pack()


# run first time

root = Tk()
root.geometry("800x600")
app = App(root)

root.mainloop()
# cap.release()
cv2.destroyAllWindows()
