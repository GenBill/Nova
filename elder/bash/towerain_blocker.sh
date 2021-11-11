CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21200 multar-adv-train.py >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 21201 multar-adv-train.py >/dev/null 2>&1 &
