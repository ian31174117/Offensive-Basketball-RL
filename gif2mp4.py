from moviepy.editor import *
import os
# Path to the folder containing GIF files
folder_path = './data'

# Get a list of all GIF files in the folder
gif_files = [f for f in os.listdir(folder_path) if f.endswith('.gif')]

# Iterate through each GIF file and convert to MP4
for gif_file in gif_files:
    # Construct the full path to the input GIF file
    gif_path = os.path.join(folder_path, gif_file)
    
    # Construct the full path for the output MP4 file
    mp4_path = os.path.splitext(gif_path)[0] + '.mp4'
    
    # Load the GIF file
    gif_clip = VideoFileClip(gif_path)
    
    # Write the GIF clip to an MP4 file
    gif_clip.write_videofile(mp4_path)
    
    print(f'Converted GIF to MP4: {mp4_path}')