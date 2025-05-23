from evaluation.funcs import load_video_batch
import cv2

VIDEO_SIZE_MAX = 256

file_path = "assets/reference_examples/car-turn-24.mp4"
video_tmp = cv2.VideoCapture(file_path)
init_height = video_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT)
init_width = video_tmp.get(cv2.CAP_PROP_FRAME_WIDTH)

if init_height > init_width:
    height = VIDEO_SIZE_MAX
    width = int((height / init_height) * init_width)
else:
    width = VIDEO_SIZE_MAX
    height = int((width / init_width) * init_height)

video = load_video_batch([file_path], 1, video_size=(height, width), video_frames=-1)
print(video.shape)

# image space -> latent space
latents = model.encode_first_stage_2DAE
# cond = {"c_crossattn": [text_emb], "fps": fps}
# inverse


# debug inversion

# 1. timestep vs layer + avg layers | for isolated time frame (several examples)
# 2. time frames -- best from 1 + bounding boxes -- bboxes + time frames