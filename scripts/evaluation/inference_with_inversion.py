import argparse, os, sys, glob, yaml, math, random
import cv2
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_video_batch, load_prompts, load_idx, load_traj, load_image_batch, get_filelist, save_videos, save_videos_with_bbox
from funcs import batch_ddim_inversion, batch_ddim_sampling_freetraj
from utils.utils import instantiate_from_config

from torchvision.io import write_video


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--ref_path", type=str, default=None, help="path to the reference video")
    parser.add_argument("--prompt_ref_file", type=str, default=None, help="a text file containing reference prompts")
    parser.add_argument("--prompt_gen_file", type=str, default=None, help="a text file containing generation prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--max_size", type=int, default=512, help="maximum size (1 dimension) of the video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--ddim_edit", type=int, default=6, help="steps of ddim for edited attention")
    parser.add_argument("--idx_file", type=str, default=None, help="a index file containing many prompts")
    return parser


def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    # load reference video
    video_tmp = cv2.VideoCapture(args.ref_path)
    init_frames = int(video_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    init_height = int(video_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    init_width = int(video_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Initial shape: {init_frames} x {init_height} x {init_width}")
    
    # fix frames number to avoid OOM
    frames = min(model.temporal_length, init_frames)
    print(f"Frame cut: {frames} x {init_height} x {init_width}")
    
    assert (args.max_size % 64 == 0), "Error: size should be multiple of 64"
    if init_height > init_width:
        height = args.max_size
        width = int((height / init_height) * init_width)
        print(f"Resizing: {frames} x {height} x {width}")
        width = ((width - 1) // 64 + 1) * 64
        print(f"Aligning: {frames} x {height} x {width}")
    else:
        width = args.max_size
        height = int((width / init_width) * init_height)
        print(f"Resizing: {frames} x {height} x {width}")
        height = ((height - 1) // 64 + 1) * 64
        print(f"Aligning: {frames} x {height} x {width}")

    video = load_video_batch([args.ref_path], 1, video_size=(height, width), video_frames=frames).to(model.device)
    # B x C x F x H x W
    # [-1, 1]
    write_video("test.mp4", ((video[0].permute(1, 2, 3, 0).cpu() + 1) / 2 * 255).to(dtype=torch.uint8), fps=args.savefps)
    #save_videos(video.unsqueeze(1), os.path.join(args.savedir, 'ref'), filenames, fps=args.savefps)
    
    # video -> latents
    latents = model.encode_first_stage_2DAE(video)
    vs = video.shape
    del video
    torch.cuda.empty_cache()

    assert os.path.exists(args.prompt_ref_file), "Error: reference video prompt file NOT Found!"
    prompt_ref_list = load_prompts(args.prompt_ref_file)
    # embed text
    text_ref_emb = model.get_learned_conditioning(prompt_ref_list)
    fps = torch.tensor([args.fps]*latents.shape[0]).to(model.device).long()
    cond = {"c_crossattn": [text_ref_emb], "fps": fps}

    # inversion
    inversed = batch_ddim_inversion(
        model, cond, latents, args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs
    )

    print(vs)
    print(latents.shape)
    print(inversed.shape)







    ### latent noise shape
    #h, w = args.height // 8, args.width // 8
    ##frames = model.temporal_length if args.frames < 0 else args.frames
    #channels = model.channels
    #
    ### saving folders
    #os.makedirs(args.savedir, exist_ok=True)
    #bboxdir = os.path.join(args.savedir, "bbox")
    #os.makedirs(bboxdir, exist_ok=True)
#
    ### step 2: load data
    ### -----------------------------------------------------------------
    #assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    #prompt_list = load_prompts(args.prompt_file)
    #idx_list_rank = load_idx(args.idx_file)
    #input_traj = load_traj(args.traj_file)
    #print(prompt_list)
    #print(idx_list_rank)
    #print(input_traj)
#
    #num_samples = len(prompt_list)
    #filename_list = [f"{id+1:04d}" for id in range(num_samples)]
#
    #samples_split = num_samples // gpu_num
    #residual_tail = num_samples % gpu_num
    #print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    #indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    #if gpu_no == 0 and residual_tail != 0:
    #    indices = indices + list(range(num_samples-residual_tail, num_samples))
    #prompt_list_rank = [prompt_list[i] for i in indices]
#
    ### conditional input
    #if args.mode == "i2v":
    #    ## each video or frames dir per prompt
    #    cond_inputs = get_filelist(args.cond_input, ext='[mpj][pn][4gj]')   # '[mpj][pn][4gj]'
    #    assert len(cond_inputs) == num_samples, f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
    #    filename_list = [f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)]
    #    cond_inputs_rank = [cond_inputs[i] for i in indices]
#
    #filename_list_rank = [filename_list[i] for i in indices]
    #
    #assert len(idx_list_rank) == len(filename_list_rank), "Error: metas are not paired!"
#
#
    ### step 3: run over samples
    ### -----------------------------------------------------------------
    #start = time.time()
    #n_rounds = len(prompt_list_rank) // args.bs
    #n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    #for idx in range(0, n_rounds):
    #    print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...', flush=True)
    #    idx_s = idx*args.bs
    #    idx_e = min(idx_s+args.bs, len(prompt_list_rank))
    #    batch_size = idx_e - idx_s
    #    filenames = filename_list_rank[idx_s:idx_e]
    #    noise_shape = [batch_size, channels, frames, h, w]
    #    fps = torch.tensor([args.fps]*batch_size).to(model.device).long()
#
    #    idx_list = idx_list_rank[idx_s:idx_e][0]
    #    # print(idx_list)
#
    #    prompts = prompt_list_rank[idx_s:idx_e]
    #    if isinstance(prompts, str):
    #        prompts = [prompts]
    #    #prompts = batch_size * [""]
    #    text_emb = model.get_learned_conditioning(prompts)
#
    #    if args.mode == 'base':
    #        cond = {"c_crossattn": [text_emb], "fps": fps}
    #    elif args.mode == 'i2v':
    #        #cond_images = torch.zeros(noise_shape[0],3,224,224).to(model.device)
    #        cond_images = load_image_batch(cond_inputs_rank[idx_s:idx_e], (args.height, args.width))
    #        cond_images = cond_images.to(model.device)
    #        img_emb = model.get_image_embeds(cond_images)
    #        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    #        cond = {"c_crossattn": [imtext_cond], "fps": fps}
    #    else:
    #        raise NotImplementedError
#
    #    ## inference
    #    batch_samples = batch_ddim_sampling_freetraj(model, cond, noise_shape, args.n_samples, \
    #                                            args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, idx_list=idx_list, input_traj=input_traj, args=args, **kwargs)
    #    ## b,samples,c,t,h,w
    #    # save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)
    #    save_videos_with_bbox(batch_samples, args.savedir, bboxdir, filenames, fps=args.savefps, input_traj=input_traj)
#
    #print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    #now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)