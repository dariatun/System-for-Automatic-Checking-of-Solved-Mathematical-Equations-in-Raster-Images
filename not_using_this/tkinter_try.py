import time
import threading  # NEW
from tkinter import *

master = Tk()
master.geometry("500x500")
master.resizable(False, False)


def tryout():
    sign2.config(text="AAA")
    for x in range(5):
        print(x)
        time.sleep(1)
    sign2.config(text="BBB")
    for x in range(5):
        print(x)
        time.sleep(1)
    sign2.config(text="CCC")


def close_window():
    master.destroy()
    sys.exit()


def thread():  # NEW
    threading.Thread(target=tryout).start()  # NEW


sign1 = Label(master, text="VNA GUI").grid(pady=10, padx=10)
sign2 = Label(master, text="Choose option to continue")
sign2.grid(pady=10, padx=10, ipadx=50)
Button(master, text='Exit', command=close_window).grid(pady=10, padx=20)
butTest = Button(master, text='test', command=thread)  # Changed
butTest.grid(pady=10, padx=20)

master.mainloop()
