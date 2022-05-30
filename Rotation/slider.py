import tkinter as tk
from tkinter import ttk
from scipy.stats import truncnorm
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
import math
from threeDrotation import rotateImage, get_truncated_normal


def create_input_frame(container):

    frame = ttk.Frame(container)

    #grid layour for the input frame
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(0, weight=7)

    for widget in frame.winfo_children():
        widget.grid(padx=0, pady=5)
    
    return frame

#root window
root = tk.Tk()
root.geometry('350x150')
root.resizable(False,False)
root.title("Image Controller")

root.columnconfigure(0, weight=1)  
root.columnconfigure(1, weight=5)

input_frame = create_input_frame(root)
input_frame.grid(column=0, row=0)

#current slider value
x_current_value = tk.DoubleVar()
y_current_value = tk.DoubleVar()
z_current_value = tk.DoubleVar()


def get_current_value(current_value):
    return '{: .2f}'.format(current_value.get())

######### Choose images and translations here!! ###############
###############################################################
filename = './images/9.png'
dx = 0
dy = 0
dz = 400
f = 200
bg_colour = '#0F0F0F0F'


# load the image with no rotations
try:
    intialise
except NameError:
    initialise = 0
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    dest = rotateImage(img, 0, 0, 0, dx, dy, dz, f)
    # cv2.imwrite("./images/rotating.png", dest)

    fig, ax = plt.subplots()
    ax.set_facecolor(bg_colour)
    im = ax.imshow(img)



def update_image():
    x_angle = int(float(get_current_value(x_current_value)))
    y_angle = int(float(get_current_value(y_current_value)))
    z_angle = int(float(get_current_value(z_current_value)))
    # print("x: ",x_angle," y: ",y_angle, "z: ",z_angle)

    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    dest = rotateImage(img, x_angle, y_angle, z_angle, dx, dy, dz, f)
    # cv2.imwrite("./images/rotating.png", dest)

    dest = cv2.cvtColor(dest,cv2.COLOR_RGBA2BGRA)
    im.set_data(dest)
    plt.draw(), plt.pause(1e-3)


def x_slider_changed(event):
    x_value_label.configure(text=get_current_value(x_current_value))
    update_image()
    

def y_slider_changed(event):
    y_value_label.configure(text=get_current_value(y_current_value))
    update_image()
        

def z_slider_changed(event):
    update_image()


#label for slider
x_slider_label = ttk.Label(
    input_frame,
    text='x-axis:'
)

x_slider_label.grid(
    column=0,
    row=0,
    sticky='w'
)

y_slider_label = ttk.Label(
    input_frame,
    text='y-axis:'
)

y_slider_label.grid(
    column=0,
    row=1,
    sticky='w'
)

z_slider_label = ttk.Label(
    input_frame,
    text='z-axis:'
)

z_slider_label.grid(
    column=0,
    row=2,
    sticky='w'
)


#slider 

def create_slider(slider, row, slider_changed, current_value):
    slider = ttk.Scale(
        input_frame,
        from_=-180, # min value
        to=180,  # max value
        orient='horizontal', #vertical
        command=slider_changed,
        variable=current_value
    )
    slider.grid(
        column=1,
        row=row,
        sticky='we'
    )
    return slider

x_slider = 0
y_slider = 0
z_slider = 0
create_slider(x_slider, 0, x_slider_changed, x_current_value)
create_slider(y_slider, 1, y_slider_changed, y_current_value)
create_slider(z_slider, 2, z_slider_changed, z_current_value)



#current value label
def create_current_value_label(current_value_label, row):
    current_value_label = ttk.Label(
        input_frame,
        text='Current Value:'
    )

    current_value_label.grid(
        row=row,
        column=3,
        columnspan=2,
        sticky='n',
        ipadx=10,
        ipady=10
    )
    return current_value_label

x_current_value_label = ttk.Label
create_current_value_label(x_current_value_label, 0)
y_current_value_label = ttk.Label
create_current_value_label(y_current_value_label, 1)
z_current_value_label = ttk.Label
create_current_value_label(z_current_value_label, 2)

#value labels
x_value_label = ttk.Label(
        input_frame,
        text=get_current_value(x_current_value)
    )

x_value_label.grid(
    row=0,
    column=5,
    columnspan=2,
    sticky='n',
    ipady=10
)

y_value_label = ttk.Label(
        input_frame,
        text=get_current_value(y_current_value)
    )

y_value_label.grid(
    row=1,
    column=5,
    columnspan=2,
    sticky='n',
    ipady=10
)

z_value_label = ttk.Label(
        input_frame,
        text=get_current_value(z_current_value)
    )

z_value_label.grid(
    row=2,
    column=5,
    columnspan=2,
    sticky='n',
    ipady=10
)


root.mainloop()