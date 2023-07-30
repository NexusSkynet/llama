from dataclasses import dataclass, field,asdict




@dataclass
class Config_Model:
    dim : int 
    n_heads : int
    n_layers : int
    vocab_size: int


@dataclass
class Config_Train:
    epoch : int 
    warmup_iter : int
    weight_decay : bool




if __name__ == "__main__":
    conf = Config_Model(4,5, 8900)
    print(asdict(conf))