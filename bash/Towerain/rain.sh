CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 731100 multar-adv-train.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 731101 multar-adv-train-1.py >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 731102 multar-adv-train-2.py >/dev/null 2>&1
