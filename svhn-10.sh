
# Test
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14439 adv-Finaltest-cifar10.py
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14440 adv-Finaltest-svhn.py


