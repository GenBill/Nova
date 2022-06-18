# CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce0.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce1.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce2.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce3.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce4.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce5.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce6.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce7.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce8.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce9.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-train-Uce10.py

CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10368 adv-Finaltest-cifar10_Uce.py
