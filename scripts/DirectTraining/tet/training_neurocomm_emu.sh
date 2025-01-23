#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=200 \
        base.gpu_id=\'0\' \
        base.seed=120 \
        base.port=\'12345\' \
        base.data=\'dvs128-gesture\' \
        base.model=\'vggsnn-neurocomm-emu\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-gesture-10-X\'\
        base.checkpoint_save=False \
        base.checkpoint_path=none \
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'none\' \
        snn-train.loss='tet' \
        snn-train.multistep=True \
        snn-train.add_time_dim=False \
        snn-train.T=4 \
        snn-train.alpha=0.00 \
        snn-train.E=-40 \
        \
        neuron_params.vthr=1.0 \
        neuron_params.num_bit=2 \
        neuron_params.modulation='analog' \
        neuron_params.num_ofdma=5 \
        neuron_params.tau=0.5 \
        neuron_params.mem_init=0.0 \
        neuron_params.multistep=True \
        neuron_params.reset_mode='hard' 