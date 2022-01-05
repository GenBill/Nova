# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10246 frost-adv-train-10-MMC.py &
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10248 frost-adv-train-10-MMC-un.py
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10244 frost-adv-train-10-MMC-none.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10241 frost-adv-train-10-vu.py &
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10242 frost-adv-train-10-vt.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10243 frost-adv-train-10-wu.py &
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10244 frost-adv-train-10-wt.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10252 adv-Finaltest-cifar10-Frost.py 

