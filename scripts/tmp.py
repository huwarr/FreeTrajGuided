from funcs import load_video_batch


video = load_video_batch(["assets/reference_examples/car-turn-24.mp4"], 1, video_frames=-1)
print(video.shape)