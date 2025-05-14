<h2>Transliteration using Recurrent Neural Networks</h2>

[Link to the wandb.ai report](https://wandb.ai/cs22m088-iit-madras/DA6401_Assignment_3/reports/DA6401-Assignment-3--VmlldzoxMjc2NzEwMQ?accessToken=u1nhkunxydtxbfbzbd64uasu6k5wwuk07njcpokrlghp1ad8ftv0zmlswbpnem50)

**Usage**

* Without Attention

```
usage: python main.py 
               [-h] [--epochs EPOCHS] 
               [--optimizer OPTIMIZER]
               [--Cell_Type CELL_TYPE] [--l_rate L_RATE] [--loss LOSS]
               [--batch_size BATCH_SIZE] [--embedding_size EMBEDDING_SIZE]
               [--hidden_layer_size HIDDEN_LAYER_SIZE]
               [--n_enc_dec_layers N_ENC_DEC_LAYERS] [--beam_size BEAM_SIZE]
               [--dropout DROPOUT]
```

* With Attention

```
usage: python attention_main.py 
               [-h] [--epochs EPOCHS] [--optimizer OPTIMIZER]
               [--Cell_Type CELL_TYPE] [--l_rate L_RATE]
               [--loss LOSS] [--batch_size BATCH_SIZE]
               [--embedding_size EMBEDDING_SIZE]
               [--hidden_layer_size HIDDEN_LAYER_SIZE]
               [--n_enc_dec_layers N_ENC_DEC_LAYERS]
               [--beam_size BEAM_SIZE] [--dropout DROPOUT]

```



Hyperparameter | Values/Usage
-------------------- | --------------------
epochs | 5, 10, 20
optimizer | "Adam", "Nadam"
Cell_Type | "RNN", "LSTM", "GRU"
loss | "categorical_crossentropy"
batch_size | 64, 32
dropout | 0.1, 0.2, 0.4
l_rate | 0.01, 0.001
embedding_size | 
hidden_layer_size | 64, 128, 256
n_enc_dec_layers | 1, 2, 3
beam_size | 1, 5, 10


**To run the code**

1. Set the hyperparameter configuration in run.sh file
2. Run `bash run.sh`


**Files**

* run.sh -- to run the code
* main.py -- main funtion (with attention)
* attention_main.py -- main funtion (without attention)
* LoadData.py -- to load the data
* Accuracy.py -- to calculate accuracy
* attention.py -- for attention layer (Bahdanau Attention)
* CreateModel.py -- for creating model architecture
* DecodeText.py -- for inference
* EncoderDecoderModel -- to get encoder and decoder model for inference
* InferAttention.py -- to plot attention heatmaps
* config.py -- to set up configuration
* sweep.yaml -- for configuring hyperparameters to run sweeps (without attention)
* attention_sweep.yaml -- for configuring hyperparameters to run sweeps (with attention)
