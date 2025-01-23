
from snncutoff.constrs.ann import *
from snncutoff.constrs.snn import *
from snncutoff.utils import add_ann_constraints, add_snn_layers
from snncutoff.models.vgg_neurocomm import VGG_NeuroComm
from .get_constrs import get_constrs
from .get_regularizer import get_regularizer
from snncutoff.utils import yaml2model
from types import SimpleNamespace

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    else:
        return ns
    
def get_model(args):
    input_size  = InputSize(args.data.lower())
    num_classes  = OuputSize(args.data.lower())
    if args.method !='ann' and args.method !='snn':
        AssertionError('Training method is wrong!')

    if args.method=='ann':
        multistep = args.multistep_ann
        model = ann_models(args.model, input_size, num_classes,multistep,args=args)
        model = add_ann_constraints(model, args.T, args.L, args.multistep_ann,
                                    ann_constrs=get_constrs(args.ann_constrs.lower(),args.method), 
                                    regularizer=get_regularizer(args.regularizer.lower(),args.method))    
        return model
    elif args.method=='snn':
        model = ann_models(args.model,input_size,num_classes,multistep=True,args=args) if args.arch_conversion else snn_models(args.model,args.T,input_size, num_classes) 
        model = add_snn_layers(model, args.T,
                                snn_layers=get_constrs(args.snn_layers.lower(),args.method), 
                                TEBN=args.TEBN,
                                neuron_params=args.neuron_params,
                                regularizer=get_regularizer(args.regularizer.lower(),args.method),
                                arch_conversion=args.arch_conversion,
                                )  
        return model
    else:
        NameError("The dataset name is not support!")
        exit(0)

def get_basemodel(name):
    if name.lower() in ['vgg11','vgg13','vgg16','vgg19',]:
        return 'vgg'
    elif name.lower() in ['resnet18','resnet20','resnet34','resnet50','resnet101','resnet152']:
        return 'resnet'
    elif name.lower() in ['sew_resnet18','sew_resnet20','sew_resnet34','sew_resnet50','sew_resnet101','sew_resnet152']:
        return 'sew_resnet'
    else:
        pass

def ann_models( model_name, input_size, num_classes,multistep,args):
    base_model = get_basemodel(model_name)
    if model_name.lower() == 'vggsnn-neurocomm':
        return VGG_NeuroComm(num_classes=num_classes,E=args.E,neuron_params=args.neuron_params)
    elif model_name.lower() == 'vggsnn-neurocomm-eval':
        from snncutoff.models.vgg_neurocomm_eval_power import VGG_NeuroComm_eval
        return VGG_NeuroComm_eval(num_classes=num_classes,E=args.E,neuron_params=args.neuron_params)
    elif model_name.lower() == 'vggsnn-neurocomm-emu':
        from snncutoff.models.vgg_neurocomm_emu import VGG_NeuroComm_emu
        return VGG_NeuroComm_emu(num_classes=num_classes,E=args.E,neuron_params=args.neuron_params)
    else:
        AssertionError('The network is not suported!')
        exit(0)

def snn_models(model_name, T, input_size, num_classes):
    base_model = get_basemodel(model_name)
    if model_name.lower() == 'vggsnn-neurocomm':
        return VGG_NeuroComm(num_classes=num_classes)
    else:
        AssertionError('This architecture is not suported yet!')

def InputSize(name):
    if 'cifar10-dvs' in name.lower() or 'dvs128-gesture' in name.lower():
        return 128 #'2-128-128'
    elif 'cifar10' in name.lower() or 'cifar100' in name.lower():
        return 32 #'3-32-32'
    elif 'imagenet' in name.lower():
        if 'tiny-imagenet' == name.lower():
            return 64
        else:
            return 224 #'3-224-224'
    elif  'ncaltech101' in name.lower():
        return 240 #'2-240-180'
    else:
        NameError('This dataset name is not supported!')

def OuputSize(name):
    if 'cifar10-dvs' == name.lower() or 'cifar10' == name.lower() :
        return 10
    elif 'dvs128-gesture' == name.lower():
        return 11
    elif 'cifar100' == name.lower():
        return 100
    elif 'ncaltech101' == name.lower():
        return 101
    elif 'imagenet-' in name.lower():
        output_size = name.lower().split("-")[-1]
        return int(output_size)
    elif 'tiny-imagenet' == name.lower():
        return 200
    elif 'imagenet' == name.lower():
        return 1000
    else:
        NameError('This dataset name is not supported!')
