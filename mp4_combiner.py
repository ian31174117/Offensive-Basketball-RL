import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, clips_array

# Path to the directory containing MP4 files
directory_path = './data'

# Get a list of all files in the directory
files = os.listdir(directory_path)

# Filter the files to include only MP4 files with the naming convention
mp4_files = [f for f in files if f.endswith('.mp4') and f.startswith('animation_game_transformer_train_')]

# Extract indices from the file names
indices = [int(f.split('_')[-1].split('.')[0]) for f in mp4_files]

# Initialize a list to store the VideoFileClip objects
clips = []

# Iterate through each index and load the corresponding MP4 files
for idx in indices:
    print("idx: " , idx)
    mp4_transformer_path = f"./data/animation_game_transformer_train_{idx}.mp4"
    mp4_real_path = f"./data/animation_game_real_train_{idx}.mp4"
    clip_real = VideoFileClip(mp4_real_path)
    clip_transformer = VideoFileClip(mp4_transformer_path)
    combined = clips_array([[clip_real, clip_transformer]])
    clips.append(combined)

final_clip = concatenate_videoclips(clips, method='compose')

# Export the final concatenated clip as a new MP4 file
final_clip.write_videofile('./data/combined_videos.mp4')

# Close all clips to free up resources
for clip in clips:
    clip.close()
final_clip.close()