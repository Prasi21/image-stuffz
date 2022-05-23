import tkinter as tk
from tkinter import ttk
from scipy.stats import truncnorm
import numpy as np  
import matplotlib.pyplot as plt
import cv2
import math
from threeDrotation import rotateImage, get_truncated_normal


def create_input_frame(container):

    frame = ttk.Frame(container)

    #grid layour for the input frame
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(0, weight=3)

    #x axis
    ttk.Label(frame, text="Rotation around x-axis").grid(column=0, row=0, sticky=tk.W)
    

    for widget in frame.winfo_children():
        widget.grit(padx=0, pady=5)
    
    return frame


#root window
root = tk.Tk()
root.geometry('300x200')
root.resizable(False,False)
root.title("slider demo")

root.columnconfigure(0, weight=1)  
root.columnconfigure(1, weight=3)


#current slider value
current_value = tk.DoubleVar()


def get_current_value():
    return '{: .2f}'.format(current_value.get())


def slider_changed(event):
    value_label.configure(text=get_current_value())
    out=int(float(get_current_value()))
    print(out)
    img = cv2.imread('./images/9.png', cv2.IMREAD_UNCHANGED)
    dest = rotateImage(img, out, 0, 0, 0, 0, 400, 200)
    cv2.imwrite("./images/9_3drotate.png", dest)
    


#label for slider
slider_label = ttk.Label(
    root,
    text='Slider:'
)

slider_label.grid(
    column=0,
    row=0,
    sticky='w'
)


#slider 
slider = ttk.Scale(
    root,
    from_=0, # min value
    to=360,  # max value
    orient='horizontal', #vertical
    command=slider_changed,
    variable=current_value
)


slider.grid(
    column=1,
    row=0,
    sticky='we'
)


#current value label
current_value_label = ttk.Label(
    root,
    text='Current Value:'
)

current_value_label.grid(
    row=1,
    columnspan=2,
    sticky='n',
    ipadx=10,
    ipady=10
)


#value label
value_label = ttk.Label(
    root,
    text=get_current_value()
)

value_label.grid(
    row=2,
    columnspan=2,
    sticky='n'
)



root.mainloop()