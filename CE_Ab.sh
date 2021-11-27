

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10241 frost-adv-train-10-CE-vu.py &
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10242 frost-adv-train-10-CE-vt.py

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10243 frost-adv-train-10-CE-wu.py &
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10244 frost-adv-train-10-CE-wt.py


