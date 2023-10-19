import os
import argparse
import torch
import torch.distributed as dist
def main(args):
    local_rank = args.local_rank
    os.environ['RANK'] = str(2)
   
    print(local_rank, os.environ['RANK'])
    print("+++++++++++++++++++=")
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = torch.distributed.get_rank()
    print(local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    main(args)