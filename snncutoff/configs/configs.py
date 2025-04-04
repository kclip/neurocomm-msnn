from pydantic import BaseModel

class BaseConfig(BaseModel):
    workers: int
    epochs: int
    batch_size: int
    start_epoch: int
    weight_decay: float
    lr: float
    print_freq: int
    seed: int
    log: str
    project: str
    port: str
    gpu_id: str
    nprocs: int
    local_rank: int
    data: str
    model: str
    dataset_path: str
    checkpoint_save: bool
    checkpoint_path: str

class neuron_params(BaseModel):
  vthr: float 
  T: float 
  tau: float
  num_bit: int
  B: int
  num_ofdma: int
  modulation: str
  mem_init: float
  multistep: bool
  reset_mode: str


class SNNConfig(BaseModel):
    method: str
    arch_conversion: bool
    ann_constrs: str
    snn_layers: str
    TEBN: bool
    T: int
    L: int
    evaluate: bool
    loss: str
    TET: bool
    regularizer: str
    means: float
    lamb: float
    alpha: float
    multistep_ann: bool
    multistep: bool
    add_time_dim: bool
    rcs_n: float
    E: float
    
class SNNTest(BaseModel):
    epsilon: float
    decay_factor: float
    cutoff_name: str
    reset_mode: str
    model_path: str

class LoggingConfig(BaseModel):
    wandb_logging: bool
    tensorboard_logging: bool
    comet_logging: bool
    run_dir: str

class AllConfig(BaseConfig, SNNConfig,LoggingConfig):
    pass

class TestConfig(BaseConfig, SNNConfig, SNNTest):
    pass