
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21001 target-adv-train-1.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21002 target-adv-train-2.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21003 target-adv-train-3.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21004 target-adv-train-4.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21005 target-adv-train-5.py >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21101 tarsha-adv-train-1.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21102 tarsha-adv-train-2.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21103 tarsha-adv-train-3.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21104 tarsha-adv-train-4.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21105 tarsha-adv-train-5.py >/dev/null 2>&1



