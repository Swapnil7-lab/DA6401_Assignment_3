#!/bin/sh

################ Without Attention ################
# python -m wandb sweep --project "Deep-Learning-RNN" sweep.yaml

python main.py  --epochs 10 --optimizer 'Adam' --l_rate 0.01 --Cell_Type 'GRU' --batch_size 64 --embedding_size 64 --n_enc_dec_layers 3 --hidden_layer_size 256 --dropout 0.2 --beam_size 10


################ For Attention ################
# python -m wandb sweep --project "Deep-Learning-RNN" sweep_attention.yaml

python attention_main.py  --epochs 10 --optimizer 'Adam' --l_rate 0.01 --Cell_Type 'GRU' --batch_size 64 --embedding_size 64 --n_enc_dec_layers 2 --hidden_layer_size 256 --dropout 0.2 --beam_size 10
