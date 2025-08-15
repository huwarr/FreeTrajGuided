ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_freetraj_512_v2.0.yaml'

res_dir="results_soft_injection_with_features_with_init_10x10"
#ref_path="assets/reference_examples/car-turn.mp4"
name="car-roundabout-24"
ref_path="assets/reference_examples/car-roundabout-24.mp4"
prompt_ref_file="prompts/inversion/text_ref.txt"
prompt_gen_file="prompts/inversion/text.txt"
idx_ref_file="prompts/inversion/idx_ref.txt"
idx_gen_file="prompts/inversion/idx.txt"


#seeds="123 44 707 606 54 111 42 765 404 1348"
seeds="404 1348"

for s in $seeds; do
    res_dir_="${res_dir}/${name}/${s}"
    
    echo "Processing: ${res_dir_}"
    
    python3 scripts/evaluation/inference_with_inversion.py \
    --seed $s \
    --ckpt_path $ckpt \
    --config $config \
    --savedir $res_dir_ \
    --n_samples 1 \
    --bs 1 --max_size 384 \
    --unconditional_guidance_scale 12.0 \
    --ddim_steps 50 \
    --ddim_eta 0.0 \
    --ref_path $ref_path \
    --prompt_ref_file $prompt_ref_file \
    --idx_ref_file $idx_ref_file \
    --prompt_gen_file $prompt_gen_file \
    --idx_gen_file $idx_gen_file \
    --ddim_edit 6 \
    --quantile 0.9 \
    --sigma 4 \
    --size_frac 0.3
done