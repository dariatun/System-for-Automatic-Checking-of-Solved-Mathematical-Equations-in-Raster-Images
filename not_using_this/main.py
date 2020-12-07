from random import randint
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image

root = Tk()
root.wm_attributes("-topmost", 1)
root.geometry('{}x{}'.format(1100, 720))  # window size
# canvas = Canvas(root, bd=0, highlightthickness=0)
canvas = Canvas(root, width=950, height=700)


# canvas.pack()

def client_exit():
    """ Quit Button """
    exit()


def pickNote(value):
    ''' Changes noteChosen var to the note's button pressed '''
    global correctNote
    noteChosen = value
    if noteChosen == correctNote:
        print("SUCCESS !!!")
        displayRandomNote(None)
    else:
        print(" :( ", noteChosen, " :( ")


# Creates button to exit the program
quitButton = tk.Button(text="Quit", command=client_exit)
quitButton.place(x=480, y=480)

# Creates buttons for various notes
aButton = tk.Button(text="A", command=lambda *args: pickNote("A"))
aButton.config(height=3, width=9)
aButton.place(x=190, y=400)

bButton = tk.Button(text="B", command=lambda *args: pickNote("B"))
bButton.config(height=3, width=9)
bButton.place(x=280, y=400)

cButton = tk.Button(text="C", command=lambda *args: pickNote("C"))
cButton.config(height=3, width=9)
cButton.place(x=370, y=400)

dButton = tk.Button(text="D", command=lambda *args: pickNote("D"))
dButton.config(height=3, width=9)
dButton.place(x=460, y=400)

eButton = tk.Button(text="E", command=lambda *args: pickNote("E"))
eButton.config(height=3, width=9)
eButton.place(x=550, y=400)

fButton = tk.Button(text="F", command=lambda *args: pickNote("F"))
fButton.config(height=3, width=9)
fButton.place(x=640, y=400)

gButton = tk.Button(text="G", command=lambda *args: pickNote("G"))
gButton.config(height=3, width=9)
gButton.place(x=730, y=400)

noteFiles = {1: 'capture.jpg', 2: "edged.jpg", 3: 'contour.jpg', 4: 'out/capture_rec.jpg', 5: 'capture1.jpg'}
notes = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

randomNote = randint(1, 5)
path = noteFiles[randomNote]
correctNote = notes[randomNote]
img = Image.open(path)
tk_img = ImageTk.PhotoImage(img)
imageOnCanvas = canvas.create_image(130, 150, image=tk_img)  # position of image center in window
canvas.pack()


def displayRandomNote(event):
    global canvas
    global imageOnCanvas
    global tk_img
    global correctNote
    global notes
    randomNote = randint(1, 5)
    path = noteFiles[randomNote]
    correctNote = notes[randomNote]
    img = Image.open(path)
    tk_img = ImageTk.PhotoImage(img)
    canvas.itemconfig(imageOnCanvas, image=tk_img)  # change the displayed picture
    canvas.pack()

    # userResponse = input("Which note?\n           ")
    # if userResponse == correctNote:
    #     print("                      SUCCESS :) !!!")
    #     print("(switch focus)")
    # else:
    #     print("                      TRY ANOTHER ONE ...")
    #     print("(switch focus)")


# print("Switch window focus to CONSOLE to input the answer. ")
# print("Swicht window focus to IMAGE (press right arrow key for a Note)")

root.bind('<Right>', displayRandomNote)  # on right arrow key display random note

root.mainloop()
