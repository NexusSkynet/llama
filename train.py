from config import Config_Train
from dataclasses import asdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



conf = Config_Train(warmup_iter=23,epoch = 45,weight_decay=True)
print(asdict(conf))

def seed(number):
    torch.manual_seed(number)






def setup():
    dist.init_process_group("nccl")



def clean():
    dist.destroy_process_group()



iter_num = 0



while True:



    checkpoint = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        

    }


    iter_num += 1

