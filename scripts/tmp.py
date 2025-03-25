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