import torch.nn as nn
from snncutoff.neuron import *
from snncutoff.constrs.ann import PreConstrs, PostConstrs, DropoutConstrs
from snncutoff.constrs.snn import BaseLayer
from snncutoff.constrs.snn import TEBNLayer


def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

def isContainer(name):
    if 'container' in name.lower():
        return True
    return False

def addPreConstrs(name):
    if 'conv2d' == name.lower() or 'linear' == name.lower() or 'pool' in name.lower() or 'flatten' in name.lower() or 'bypass' in name.lower():
        return True
    return False

def addPostConstrs(name):
    if 'pool' in name.lower() or 'flatten' in name.lower():
        return True
    return False

def addSingleStep(name):
    if  'lifspike' in name or 'gs' in name:
        return True
    if 'constrs' in name or 'baselayer' in name:
        if  'preconstrs' in name or 'postconstrs' in name:
            return False
        else:
            return True
    return False

def multi_to_single_step(model, multistep_ann, reset_mode):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = multi_to_single_step(module,multistep_ann,reset_mode)
        if addSingleStep(module.__class__.__name__.lower()):
            model._modules[name].multistep = False
            model._modules[name].regularizer = None
        if  addPreConstrs(module.__class__.__name__.lower()) and not multistep_ann:
            model._modules[name] = PreConstrs(T=1, multistep=False, module=model._modules[name])
        if  addPostConstrs(module.__class__.__name__.lower()) and not multistep_ann:
            model._modules[name] = PostConstrs(T=1, multistep=False, module=model._modules[name]) 
        if  'preconstrs' in module.__class__.__name__.lower():
            model._modules[name].multistep=False  
        if  'postconstrs' in module.__class__.__name__.lower():
            model._modules[name].multistep=False  
    return model

def set_multistep(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = set_multistep(module)
        if hasattr(module, "multistep"):
            model._modules[name].multistep = True
    return model


def preprocess_ann_arch(model):
    model = set_multistep(model)
    model = nn.Sequential(
        *list(model.children())[1:],
        )
    return model


def set_dropout(model,p=0.0,training=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = set_dropout(module,p,training=training)
        if training:
            if  'baselayer' in module.__class__.__name__.lower():
                model._modules[name] = DropoutConstrs(module=model._modules[name],p=p)
                model._modules[name].train()
        else:
            if  'dropoutconstrs' in module.__class__.__name__.lower():
                model._modules[name] = model._modules[name].module
                model._modules[name].eval()
    return model


def _add_ann_constraints(model, T, L, multistep_ann, ann_constrs, regularizer=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_ann_constraints(module, T, L, multistep_ann, ann_constrs,regularizer)
        if  'relu' == module.__class__.__name__.lower():
            model._modules[name] = ann_constrs(T=T, L=L, regularizer=regularizer)
        if  addPreConstrs(module.__class__.__name__.lower()) and multistep_ann:
            model._modules[name] = PreConstrs(T=T, module=model._modules[name])
        if  addPostConstrs(module.__class__.__name__.lower()) and multistep_ann:
            model._modules[name] = PostConstrs(T=T, module=model._modules[name])    
    return model

def add_ann_constraints(model, T, L, multistep_ann, ann_constrs, regularizer=None):
    model = _add_ann_constraints(model, T, L, multistep_ann, ann_constrs, regularizer=regularizer)
    if multistep_ann:
        model = nn.Sequential(
            *list(model.children()),  
            PostConstrs(T=T, module=None)    # Add the new layer
            )
    else:
        model = nn.Sequential(
            PreConstrs(T=T, module=None),
            *list(model.children()),  
            PostConstrs(T=T, module=None)    # Add the new layer
            )
    return model

def addSNNLayers(name):
    if 'relu' == name.lower() or 'lifspike' == name.lower():
        return True
    return False


def _add_snn_layers(model, T, snn_layers, neuron_params=None, regularizer=None, TEBN=None, arch_conversion=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_snn_layers(module, T, snn_layers,neuron_params,regularizer, TEBN=TEBN, arch_conversion=arch_conversion)
        if  addSNNLayers(module.__class__.__name__.lower()):
            model._modules[name] = snn_layers(T=T, regularizer=regularizer,neuron_params=neuron_params)
        if  addPreConstrs(module.__class__.__name__.lower()) and arch_conversion:
            model._modules[name] = PreConstrs(T=T, module=model._modules[name])
        if  addPostConstrs(module.__class__.__name__.lower()) and arch_conversion:
            model._modules[name] = PostConstrs(T=T, module=model._modules[name])    
        if TEBN:
            if  'norm2d' in module.__class__.__name__.lower():
                model._modules[name] = TEBNLayer(T=T, num_features=model._modules[name].num_features)  
    return model

def add_snn_layers(model, T, snn_layers, TEBN=False, neuron_params=None,regularizer=None, arch_conversion=True):
    model = _add_snn_layers(model, T, snn_layers,neuron_params=neuron_params, regularizer=regularizer,TEBN=TEBN, arch_conversion=arch_conversion)
    if arch_conversion:
        model = nn.Sequential(
            *list(model.children()),  
            PostConstrs(T=T, module=None)    # Add the new layer
            ) 
    return model

def reset_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = reset_neuron(module)
        if hasattr(module, "neuron"):
            model._modules[name].neuron.reset()
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

from snncutoff.models.vgg_neurocomm import wirelss_ch
def split_model(model):
    before_layers = []
    after_layers = []
    found_custom_layer = False
    
    # Recursive function to traverse and split layers
    def _split_layers(module, before_layers, after_layers):
        nonlocal found_custom_layer
        for name, sub_module in module._modules.items():
            if found_custom_layer:
                after_layers.append(sub_module)
            else:
                if isinstance(sub_module, wirelss_ch):
                    found_custom_layer = True
                    # after_layers.append(sub_module)
                else:
                    before_layers.append(sub_module)
    
    _split_layers(model, before_layers, after_layers)
    
    # Define the part of the model before the custom layer
    class BeforeCustomLayerModel(nn.Module):
        def __init__(self, before_layers):
            super(BeforeCustomLayerModel, self).__init__()
            self.features = nn.Sequential(*before_layers)
        
        def forward(self, x):
            x = self.features(x)
            return x

    # Define the part of the model after the custom layer
    class AfterCustomLayerModel(nn.Module):
        def __init__(self, after_layers):
            super(AfterCustomLayerModel, self).__init__()
            self.features = nn.Sequential(*after_layers)
        
        def forward(self, x):
            x = self.features(x)
            return x
    
    # Instantiate the new models
    before_model = BeforeCustomLayerModel(before_layers)
    after_model = AfterCustomLayerModel(after_layers)
    
    return before_model, after_model