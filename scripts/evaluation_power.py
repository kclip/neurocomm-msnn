import os
import torch
import warnings
import torch.optim
import hydra
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from snncutoff import data_loaders
import numpy as np
from snncutoff.configs import *
from omegaconf import DictConfig
from snncutoff.Evaluator import Evaluator
from snncutoff.utils import multi_to_single_step, preprocess_ann_arch
from snncutoff.API import get_model
from snncutoff.utils import save_pickle
import torch.backends.cudnn as cudnn
from snncutoff.utils import set_seed, split_model
from snncutoff.utils import reset_neuron
from types import SimpleNamespace
import tensorflow as tf

@hydra.main(version_base=None, config_path='../snncutoff/configs', config_name='test')
def main(cfg: DictConfig):
    args = TestConfig(**cfg['base'], **cfg['snn-train'], **cfg['snn-test'])
    a =  {'neuron_params': None}
    args = SimpleNamespace(**{**args.__dict__,**a})
    args.neuron_params = cfg['neuron_params']
    if args.gpu_id != 'none':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)



    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = data_loaders.get_data_loaders(path=args.dataset_path, data=args.data, transform=False,resize=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    ############ 
    E_set=[-80,-75,-70,-65,-60,-55,-54,-53,-52,-51,-50,-49,-48,-47,-46,-45,-40,-35,-30,-25,-20] # final 100 digital 2 B
    

    acc_list = []
    for E in E_set:
        args.E = E
        models = get_model(args)
        i= 0
        path = args.model_path
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        models.load_state_dict(state_dict, strict=False)
        if not args.multistep:
            if not args.multistep_ann:
                models = preprocess_ann_arch(models)
            models = multi_to_single_step(models, args.multistep_ann, reset_mode=args.reset_mode)
        models.to(device)
        evaluator = Evaluator(models,args=args)
        acc, loss = evaluator.evaluation(test_loader)
        print(acc)
        print(np.mean(loss))
        acc_list.append(acc)
        if args.neuron_params.modulation == 'noiseless':
            break

    acc_list = np.array(acc_list)
    save_pickle(acc_list,name='final_peak_power_ofdma_'+str(args.neuron_params.num_ofdma)+'_vs_acc_'+args.neuron_params.modulation, path=os.path.dirname(path))
    # save_pickle(acc_list,name='final_ofdma_'+str(args.neuron_params.num_ofdma)+'_vs_acc_'+args.neuron_params.modulation, path=os.path.dirname(path))
  

if __name__ == '__main__':
   main()
