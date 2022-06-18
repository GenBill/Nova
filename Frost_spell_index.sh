

## Main Cifar10
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 81234 frost-adv-train-10.py

## Main Cifar100
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 81234 frost-adv-train-100.py

## Main Cifar100-pre
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 81234 frost-adv-train-100-pre.py

## Main SVHN
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 81234 frost-adv-train-svhn.py


## Main Cifar10
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port 81001 clean-train-svhn.py &
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port 81002 adv-train-svhn.py &
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 81003 clean-train-10.py &
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 81004 adv-train-10.py &


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 81004 frost-adv-train-10-fsdu.py
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 81004 frost-adv-train-10-fsdt.py

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14440 frost-adv-train-10-Udt.py &
CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14441 frost-adv-train-10-Ufsdt.py
sleep 1h; CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14442 adv-Finaltest-cifar10_copy.py

CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14441 frost-adv-train-10-Ufsdt.py &
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14440 frost-adv-train-10-Udt.py; 
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 14442 adv-Finaltest-cifar10_copy.py

