import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def visualize_game(data, anim_path = "./data/animated_game.gif", title = "Game"):
    court = plt.imread("court.png")
    def update(frame):
        if frame < data.shape[0]:
            plt.cla()
            plt.title(title)
            #the ball
            plt.scatter(data[frame][0], data[frame][1], color = "black")
            #the offensive
            plt.scatter(data[frame][[3,5,7,9,11]], data[frame][[4,6,8,10,12]], color = "red")
            #the defensive
            plt.scatter(data[frame][[13,15,17,19,21]], data[frame][[14,16,18,20,22]], color = "blue")
            plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
            plt.xlim(0,94)
            plt.ylim(0,50)
    fig, ax = plt.subplots()
    
    anim = FuncAnimation(fig, update, frames = data.shape[0], interval = 200)
    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    anim.save(anim_path, writer = "ffmpeg", fps=5)
    print(f"gif saved at {anim_path}")