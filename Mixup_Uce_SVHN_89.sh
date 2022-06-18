# CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce0.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce1.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce2.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce3.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce4.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce5.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce6.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce7.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce8.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce9.py
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10240 shades-clean-svhn-Uce10.py

CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10368 adv-Finaltest-svhn_Uce.py
