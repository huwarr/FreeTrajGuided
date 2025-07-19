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

from funcs import load_model_checkpoint, load_video_batch, load_prompts, load_idx, load_traj, load_image_batch, get_filelist, save_videos, save_videos_with_bbox, save_videos_with_bbox_and_ref
from funcs import batch_ddim_inversion, batch_ddim_sampling_freetraj, batch_ddim_sampling_freetraj_with_path
from utils.utils import instantiate_from_config
from utils.utils_freetraj import plan_path

from torchvision.io import write_video
from torchvision.transforms.functional import resize
from scipy.ndimage import gaussian_filter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt", help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_freetraj_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--ref_path", type=str, default="assets/reference_examples/car-roundabout-24.mp4", help="path to the reference video")
    parser.add_argument("--prompt_ref_file", type=str, default="prompts/inversion/text_ref.txt", help="a text file containing reference prompts")
    parser.add_argument("--prompt_gen_file", type=str, default="prompts/inversion/text.txt", help="a text file containing generation prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--max_size", type=int, default=512, help="maximum size (1 dimension) of the video")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--ddim_edit", type=int, default=6, help="steps of ddim for edited attention")
    parser.add_argument("--idx_ref_file", type=str, default="prompts/inversion/idx_ref.txt", help="a index file containing many prompts")
    parser.add_argument("--idx_gen_file", type=str, default="prompts/inversion/idx.txt", help="a index file containing many prompts")
    parser.add_argument("--quantile", type=float, default=0.85, help="quantile for binarizing cross-attention maps")
    parser.add_argument("--kernel_size", type=int, default=5, help="kernel size for binary morphology")
    parser.add_argument("--sigma", type=int, default=4, help="sigma for Gaussian smoothing")
    parser.add_argument("--size_frac", type=float, default=0.3, help="fraction of size for bbox, based on center of mass")
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
    
    init_fps = video_tmp.get(cv2.CAP_PROP_FPS)
    print(f"Initial fps: {init_fps}")
    fps_n = int(frames / (init_frames / init_fps))
    print(f"Extracted fps: {fps_n}")
    
    # assert (args.max_size % 64 == 0), "Error: size should be multiple of 64"
    # if init_height > init_width:
    #     height = args.max_size
    #     width = int((height / init_height) * init_width)
    #     print(f"Resizing: {frames} x {height} x {width}")
    #     width = ((width - 1) // 64 + 1) * 64
    #     print(f"Aligning: {frames} x {height} x {width}")
    # else:
    #     width = args.max_size
    #     height = int((width / init_width) * init_height)
    #     print(f"Resizing: {frames} x {height} x {width}")
    #     height = ((height - 1) // 64 + 1) * 64
    #     print(f"Aligning: {frames} x {height} x {width}")
    height = 320
    width = 512

    video = load_video_batch([args.ref_path], 1, video_size=(height, width), video_frames=frames).to(model.device)
    # B x C x F x H x W
    # [-1, 1]
    write_video("test.mp4", ((video[0].permute(1, 2, 3, 0).cpu() + 1) / 2 * 255).to(dtype=torch.uint8), fps=args.savefps)
    #save_videos(video.unsqueeze(1), os.path.join(args.savedir, 'ref'), filenames, fps=args.savefps)
    
    # video -> latents
    latents = model.encode_first_stage_2DAE(video)
    video = video.cpu()
    #del video
    torch.cuda.empty_cache()

    assert os.path.exists(args.prompt_ref_file), "Error: reference video prompt file NOT Found!"
    prompt_ref_list = load_prompts(args.prompt_ref_file)
    # embed text
    text_ref_emb = model.get_learned_conditioning(prompt_ref_list)
    fps = torch.tensor([fps_n]*latents.shape[0]).to(model.device).long()
    cond = {"c_crossattn": [text_ref_emb], "fps": fps}

    
    
    
    # inversion
    inversed, intermediates = batch_ddim_inversion(
        model, cond, latents, args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, log_every_t=1, return_cross_attn=True, **kwargs
    )
    # filter cross attention maps
    cmaps = []
    for i in range(len(intermediates['cmaps'])):
        cmaps_curr = intermediates['cmaps'][i]
        cmaps.append([])
        for j in range(len(cmaps_curr)):
            if len(cmaps_curr[j]) > 0:
                cmaps[-1].append(cmaps_curr[j][0][0])

    # Get token index
    idx_list_ref = load_idx(args.idx_ref_file) # [[i]]
    ind_ref = idx_list_ref[0][0]

    # trajectory for FreeTraj: List([h_start, h_end, w_start, w_end], ...)
    paths = []
    input_traj = []

    n_layers = len(cmaps[0])
    cmaps_frames = []
    for frame in tqdm(range(frames), desc="Building trajectory from cross-attention maps"):
        # Compute average cross-attention map for given frame and provided token index
        cmaps_l = []
        for i in range(args.ddim_steps):
            cmaps_t = []
            for j in range(n_layers):
                # select timestep
                cmaps_curr = cmaps[i]
                # select layer
                cmap = cmaps_curr[j]
                # select word, +1 for <start_of_text>
                cmap = cmap[..., ind_ref+1]
                # reshape to [C, F, H, W]
                sz = int((cmap.shape[-1] / (inversed.shape[-1] / inversed.shape[-2])) ** (1/2))
                cmap = cmap.reshape(-1, frames, sz, int(sz * width / height))
                # average over C
                cmap = cmap.mean(dim=0)[frame].unsqueeze(0)
                
                if len(cmaps_t) == 0:
                    cmaps_t.append(cmap)
                else:
                    cmap = resize(cmap, cmaps_t[0].shape[1:])
                    cmaps_t.append(cmap)
            cmap = torch.stack(cmaps_t).mean(0)
            cmaps_l.append(cmap)
        cmap = torch.stack(cmaps_l).mean(0)
        cmap = cmap.squeeze(0)
        
        # gaussian smoothing
        cmap = gaussian_filter(cmap, args.sigma)

        # min max normalize
        cmap_minmax = (cmap - cmap.min()) / (cmap.max() - cmap.min())
        cmaps_frames.append(cmap_minmax)

        # binarize
        cmap = cmap.numpy()
        thresh = np.quantile(cmap, args.quantile)
        cmap = (cmap > thresh).astype(np.uint8)

        # # remove noise with binary morphology (opening)
        # kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
        # cmap = cv2.morphologyEx(cmap.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # compute bbox
        x,y,w,h = cv2.boundingRect(cv2.findNonZero(cmap))
        # x,y,w,h -> h_start,h_end,w_start,w_end
        h_start = y
        h_end = y + h
        w_start = x
        w_end = x + w
        # get relative coords
        hh, ww = cmap.shape
        h_start /= hh
        h_end /= hh
        w_start /= ww
        w_end /= ww
        
        # # center of mass
        # cmap = cmap.numpy()
        # total_mass = np.sum(cmap)
        # rows, cols = cmap.shape
        # y_coords, x_coords = np.indices((rows, cols))
        # weighted_sum_x = np.sum(x_coords * cmap)
        # weighted_sum_y = np.sum(y_coords * cmap)
        # center_of_mass_x = weighted_sum_x / total_mass
        # center_of_mass_y = weighted_sum_y / total_mass 
        
        # # compute bbox
        # hh, ww = cmap.shape
        # bbox_w_half = ww * args.size_frac / 2
        # bbox_h_half = hh * args.size_frac / 2
        # h_start = center_of_mass_y - bbox_h_half
        # h_end = center_of_mass_y + bbox_h_half
        # w_start = center_of_mass_x - bbox_w_half
        # w_end = center_of_mass_x + bbox_w_half
        # # get relative coords
        # h_start /= hh
        # h_end /= hh
        # w_start /= ww
        # w_end /= ww

        # add to paths
        paths.append([h_start, h_end, w_start, w_end])

        if frame in [0, 5, 10, 15]:
            input_traj.append([frame, h_start, h_end, w_start, w_end])

    del inversed
    del intermediates
    torch.cuda.empty_cache()
    
    
    

    # ----- FreeTraj starts here

    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)
    bboxdir = os.path.join(args.savedir, "bbox")
    os.makedirs(bboxdir, exist_ok=True)
    refdir = os.path.join(args.savedir, "reference")
    os.makedirs(refdir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_gen_file), "Error: generation prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_gen_file)
    idx_list_rank = load_idx(args.idx_gen_file)
    #input_traj = load_traj(args.traj_file)
    print(prompt_list)
    print(idx_list_rank)
    #print(input_traj)

    num_samples = len(prompt_list)
    filename_list = [f"{id+1:04d}" for id in range(num_samples)]

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]
    
    assert len(idx_list_rank) == len(filename_list_rank), "Error: metas are not paired!"

    del cond
    del fps 
    del text_ref_emb
    torch.cuda.empty_cache()

    ## latent noise shape
    _, channels, frames, h, w = latents.shape
    
    # print(paths)
    # print(input_traj)
    
    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in range(0, n_rounds):
        print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...', flush=True)
        idx_s = idx*args.bs
        idx_e = min(idx_s+args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([fps_n]*batch_size).to(model.device).long()

        idx_list = idx_list_rank[idx_s:idx_e][0]
        # print(idx_list)

        prompts = prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, str):
            prompts = [prompts]
        #prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompts)

        cond = {"c_crossattn": [text_emb], "fps": fps}
        
        ## inference
        #batch_samples = batch_ddim_sampling_freetraj(model, cond, noise_shape, args.n_samples, \
        #                                        args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, idx_list=idx_list, input_traj=input_traj, args=args, **kwargs)
        batch_samples = batch_ddim_sampling_freetraj_with_path(model, cond, noise_shape, args.n_samples, \
                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, idx_list=idx_list, paths=paths, args=args, cmap=cmaps_frames, **kwargs)
        ## b,samples,c,t,h,w
        #save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)
        paths_ = plan_path(input_traj)
        save_videos_with_bbox_and_ref(video, batch_samples, args.savedir, bboxdir, refdir, filenames, fps=args.savefps, paths=paths)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")
    



if __name__ == '__main__':
    #now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)