# From no_2077
# CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12016 clean-train.py >/dev/null 2>&1
# CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12018 adv-train.py >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17022 shades-clean-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17024 shades-adv-train.py >/dev/null 2>&1
# CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 12026 double-adv-train.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17028 mesa-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17032 target-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17032 tarsha-adv-train.py >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 17036 adv-test-cifar100.py
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port 17036 adv-test-cifar100.py
