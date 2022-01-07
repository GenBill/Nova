# Ours
CUDA_VISIBLE_DEVICES=6,7,8,9 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14438 frost-adv-train-100.py &

# Baseline
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14439 clean-train-100.py
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14440 adv-train-100.py


