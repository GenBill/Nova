CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 131000 target-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 131001 target-adv-train-1.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 131002 target-adv-train-2.py >/dev/null 2>&1
