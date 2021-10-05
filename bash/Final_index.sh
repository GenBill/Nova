CUDA_VISIBLE_DEVICES=5 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 21204 clean-train.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 21205 adv-train.py >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 212061 multar-adv-train.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 212062 multar-adv-train.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 21207 target-adv-train.py >/dev/null 2>&1 &


