CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10246 frost-adv-train-10-MMC.py &
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10248 frost-adv-train-10-MMC-un.py


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10241 frost-adv-train-10-CE-vu.py &
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10242 frost-adv-train-10-CE-vt.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10243 frost-adv-train-10-CE-wu.py &
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10244 frost-adv-train-10-CE-wt.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10252 adv-Finaltest-cifar10-Frost.py 


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10253 frost-adv-train-100-fsdt.py
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10254 frost-adv-train-100-fsdu.py


CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10253 adv-Finaltest-cifar10_copy.py
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10254 adv-Finaltest-svhn.py

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10255 adv-Finaltest-cifar10_trans_FS.py
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10256 adv-Finaltest-svhn_trans_FS.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10256 frost-adv-train-10-dt.py; 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10257 adv-Finaltest-cifar10_copy.py

CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --nproc_per_node=1 --master_port 10257 frost-adv-train-cubFS.py