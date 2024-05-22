import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import imageio.v2 as imageio
from PIL import Image

def visualize_game(data, anim_path = "./data/animated_game.gif", title = "Game", show_trail = True):
    court = plt.imread("court.png")
    final_court = plt.imread("court.png")#record the trail
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
            if show_trail:
                for i in range(frame,frame - 10,-1):
                    if i>=0:
                        plt.scatter(data[i][[13,15,17,19,21]], data[i][[14,16,18,20,22]], color = "blue", alpha = 0.1)
            plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
            plt.xlim(0,94)
            plt.ylim(0,50)
    fig, ax = plt.subplots()

    anim = FuncAnimation(fig, update, frames = data.shape[0], interval = 200)
    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])

    anim.save(anim_path, writer = "ffmpeg", fps=5)
    print(f"gif saved at {anim_path}")

    plt.cla()
    for i in range(data.shape[0]):
        #show the defensive trail in 1 png
        plt.title(title)
        #the defensive
        plt.scatter(data[i][[13]], data[i][[14]], color = "b", alpha = 0.25)
        plt.scatter(data[i][[15]], data[i][[16]], color = "g", alpha = 0.25)
        plt.scatter(data[i][[17]], data[i][[18]], color = "r", alpha = 0.25)
        plt.scatter(data[i][[19]], data[i][[18]], color = "c", alpha = 0.25)
        plt.scatter(data[i][[21]], data[i][[18]], color = "m", alpha = 0.25)
        #the offensive
        plt.scatter(data[i][[3]], data[i][[4]], color = "black", alpha = 0.1)
        plt.scatter(data[i][[5]], data[i][[6]], color = "black", alpha = 0.1)
        plt.scatter(data[i][[7]], data[i][[8]], color = "black", alpha = 0.1)
        plt.scatter(data[i][[9]], data[i][[10]], color = "black", alpha = 0.1)
        plt.scatter(data[i][[11]], data[i][[12]], color = "black", alpha = 0.1)
    plt.imshow(final_court, zorder=0, extent=[0, 100 - 6, 50, 0])
    plt.xlim(0,94)
    plt.ylim(0,50)
    plt.savefig(anim_path.replace(".gif",".png"))

def visualize_game_5images(data, anim_path = "./data/animated_game.gif", title = "Game", show_trail = True):
    helper(data[0:len(data)//5], anim_path.replace(".gif","_1.gif"), title)
    helper(data[len(data)//5:2*len(data)//5], anim_path.replace(".gif","_2.gif"), title)
    helper(data[2*len(data)//5:3*len(data)//5], anim_path.replace(".gif","_3.gif"), title)
    helper(data[3*len(data)//5:4*len(data)//5], anim_path.replace(".gif","_4.gif"), title)
    helper(data[4*len(data)//5:], anim_path.replace(".gif","_5.gif"), title)
    return
def helper(data, anim_path, title = "Game"):
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
            for i in range(frame,0,-1):
                if i>=0:
                    plt.scatter(data[i][[13,15,17,19,21]], data[i][[14,16,18,20,22]], color = "blue", alpha = 0.5)
            plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
            plt.xlim(0,94)
            plt.ylim(0,50)
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, update, frames = data.shape[0], interval = 60)
    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])

    anim.save(anim_path, writer = "ffmpeg", fps=5)
    print(f"gif saved at {anim_path}")
    return

def visualize_aligned_20_gifs(data, anim_path = "./data/animated_game.gif", title = "Game", show_trail = True):
    court = plt.imread("court.png")
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    
    print("data", data[0][0])
    def update(frame):
        idx_20 = frame // data[0][0].shape[0]
        idx_frame = frame % data[0][0].shape[0]
        if idx_frame <data[idx_20][0].shape(0):
            plt.cla()
            plt.title(title)
            #the ball of generated
            axs[0].scatter(data[idx_20][0][idx_frame][0], data[idx_20][0][idx_frame][1], color = "black")
            #the offensive of generated
            axs[0].scatter(data[idx_20][0][idx_frame][[3,5,7,9,11]], data[idx_20][0][idx_frame][[4,6,8,10,12]], color = "red")
            #the defensive of generated
            axs[0].scatter(data[idx_20][0][idx_frame][[13,15,17,19,21]], data[idx_20][0][idx_frame][[14,16,18,20,22]], color = "blue")
            #the ball of real
            axs[1].scatter(data[idx_20][1][idx_frame][0], data[idx_20][1][idx_frame][1], color = "black")
            #the offensive of real
            axs[1].scatter(data[idx_20][1][idx_frame][[3,5,7,9,11]], data[idx_20][1][idx_frame][[4,6,8,10,12]], color = "red")
            #the defensive of real
            axs[1].scatter(data[idx_20][1][idx_frame][[13,15,17,19,21]], data[idx_20][1][idx_frame][[14,16,18,20,22]], color = "blue")
            axs[0].set_xlim(0,94)
            axs[0].set_ylim(0,50)
            axs[1].set_xlim(0,94)
            axs[1].set_ylim(0,50)
    anim = FuncAnimation(fig, update, frames = data[0][0].shape[0] * 20, interval = 60 * 20)
    anim.save(anim_path, writer = "ffmpeg", fps=5)
