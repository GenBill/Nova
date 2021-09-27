
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 21201 multar-adv-train.py

CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21200 multar-adv-train.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3,4 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21201 multar-adv-train-1.py >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=5,6 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21202 multar-adv-train-2.py >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21203 multar-adv-train-3.py >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21204 multar-adv-train-4.py >/dev/null 2>&1
