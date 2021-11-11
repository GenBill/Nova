# From no_2077
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12016 clean-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12022 shades-clean-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12024 shades-adv-train.py >/dev/null 2>&1
# CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12026 double-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12028 mesa-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12030 adv-train.py >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12032 target-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12034 tarsha-adv-train.py >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12038 adv-test-cifar10.py
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12088 detector.py