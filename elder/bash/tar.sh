CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17312 target-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17316 tarsha-adv-train.py >/dev/null 2>&1

