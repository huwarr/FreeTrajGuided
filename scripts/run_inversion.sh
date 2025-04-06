name="inversion_test"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_freetraj_512_v2.0.yaml'

res_dir="results_inversion"
ref_path="assets/reference_examples/car-roundabout-24.mp4"
prompt_ref_file="prompts/inversion/text_ref.txt"
prompt_gen_file="prompts/inversion/text.txt"
idx_file="prompts/inversion/idx.txt"


python3 scripts/evaluation/inference_with_inversion.py \
--seed 123 \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --max_size 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 0.0 \
--ref_path $ref_path \
--prompt_ref_file $prompt_ref_file \
--prompt_gen_file $prompt_gen_file \
--idx_file $idx_file \
--ddim_edit 6 \
--fps 16
