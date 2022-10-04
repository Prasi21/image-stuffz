from scipy.stats import truncnorm
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
import math
import sys
from threeDrotation import rotateImage, get_truncated_normal

filename = './images/9.png'
dx = 0
dy = 0
dz = 400
f = 200
bg_colour = '#0F0F0F0F'
amp = 180
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)


fig, ax = plt.subplots()
ax.set_facecolor(bg_colour)
im = ax.imshow(img)

# fig = plt.figure()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)


# for t in range(0,1000,1):
def loopAnim():
    print("Ctr + c to exit")
    t = 0
    x_arr = [0, 0]
    y_arr = [0, 0]
    z_arr = [0, 0]
    while(True):
        animate1(t, x_arr, y_arr, z_arr)
        t += 1


def animate1(t, x_arr, y_arr, z_arr):    
    x = x_arr[0]
    y = y_arr[0]
    z = z_arr[0]
    x = amp*np.sin(1.4*np.radians(t))
    y = amp*np.sin(2.932*np.radians(t))
    z = amp*np.sin(3.1*np.radians(t))
    myStr = "\rt: "+str(t)+" x: "+str(x)+ " y: "+str(y)+" z: "+str(z)
    sys.stdout.write(f'{myStr:>100}')
    sys.stdout.flush()


    dst = rotateImage(img, x, y, z, dx, dy, dz, f)
    dst = cv2.cvtColor(dst,cv2.COLOR_RGBA2BGRA)
    im.set_data(dst)
    plt.draw(), plt.pause(1e-3)
    getNextVal(x_arr)
    getNextVal(y_arr)
    getNextVal(z_arr)

def getNextVal(numList):
    cur, target = numList[0], numList[1]

    if(abs(cur - target) < 30):
        target = np.random.randint(-720,720)
    elif(cur>target):
        cur -= np.random.randint(5,10)
    else:
        cur += np.random.randint(1,5)
    numList[0] = cur
    numList[1] = target


def init():
    # line.set_data([], [])
    im.set_data(img)
    return im,

def animate2(t):
    x = int(amp*np.sin(1*np.radians(t)))
    y = int(amp*np.sin(2*np.radians(t)))
    z = int(amp*np.sin(3*np.radians(t)))
    dst = rotateImage(img, x, y, z, dx, dy, dz, f)
    dst = cv2.cvtColor(dst,cv2.COLOR_RGBA2BGRA)
    im.set_data(dst)

# anim = animation.FuncAnimation(fig, animate2, init_func=init,
#                                frames=200, interval=20, blit=True)
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# plt.show()

loopAnim()