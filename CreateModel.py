import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, GRU, Concatenate, TimeDistributed
from tqdm import tqdm
import pdb
from attention import AttentionLayer

def BuildModel(Cell_Type, n_enc_dec_layers, hidden_layer_size, num_encoder_tokens, num_decoder_tokens, dropout, emb_size):
	# Encoder
	encoder_inputs = keras.Input(shape=(None,), name="enc_input")
	enc_emb =  Embedding(num_encoder_tokens, emb_size, name="enc_embedding")(encoder_inputs)

	if Cell_Type == 'RNN':
		if n_enc_dec_layers == 1:	
			encoder_outputs1, h1 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_states = [h1]
		elif n_enc_dec_layers == 2:
			encoder_outputs1, h1 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_states = [h1, h2]
		elif n_enc_dec_layers == 3:
			encoder_outputs1, h1 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_outputs3, h3 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer3")(encoder_outputs2)
			encoder_states = [h1, h2, h3]

	elif Cell_Type == 'LSTM':
		if n_enc_dec_layers == 1:
			encoder_outputs1, h1, c1 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_states = [h1, c1]
		elif n_enc_dec_layers == 2:
			encoder_outputs1, h1, c1 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1") (enc_emb)
			encoder_outputs2, h2, c2 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_states = [h1, c1, h2, c2]
		elif n_enc_dec_layers == 3:
			encoder_outputs1, h1, c1 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2, c2 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_outputs3, h3, c3 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer3")(encoder_outputs2)
			encoder_states = [h1, c1, h2, c2, h3, c3]
	elif Cell_Type == 'GRU':
		if n_enc_dec_layers == 1:
			encoder_outputs1, h1 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_states = [h1]
		elif n_enc_dec_layers == 2:
			encoder_outputs1, h1 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_states = [h1, h2]
		elif n_enc_dec_layers == 3:
			encoder_outputs1, h1 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_outputs3, h3 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer3")(encoder_outputs2)
			encoder_states = [h1, h2, h3]

	# Decoder
	decoder_inputs = keras.Input(shape=(None,), name="dec_input")
	dec_emb_layer = Embedding(num_decoder_tokens, hidden_layer_size, name="dec_embedding")
	dec_emb = dec_emb_layer(decoder_inputs)

	if Cell_Type == 'RNN':
		if n_enc_dec_layers == 1:
			decoder_outputs1, dh1 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
		elif n_enc_dec_layers == 2:
			decoder_outputs1, dh1 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
		elif n_enc_dec_layers == 3:
			decoder_outputs1, dh1 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
			decoder_outputs3, dh3 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer3")(decoder_outputs2, initial_state=[h3])
		
	elif Cell_Type == 'LSTM':
		if n_enc_dec_layers == 1:
			decoder_outputs1, dh1, dc1 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1, c1])
		elif n_enc_dec_layers == 2:
			decoder_outputs1, dh1, dc1 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1, c1])
			decoder_outputs2, dh2, dc2 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2, c2])
		elif n_enc_dec_layers == 3:
			decoder_outputs1, dh1, dc1 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1, c1])
			decoder_outputs2, dh2, dc2 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2, c2])
			decoder_outputs3, dh3, dc3 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer3")(decoder_outputs2, initial_state=[h3, c3])
	elif Cell_Type == 'GRU':
		if n_enc_dec_layers == 1:
			decoder_outputs1, dh1 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
		elif n_enc_dec_layers == 2:
			decoder_outputs1, dh1 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
		elif n_enc_dec_layers == 3:
			decoder_outputs1, dh1 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
			decoder_outputs3, dh3 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer3")(decoder_outputs2, initial_state=[h3])

	decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="dense_layer")

	if n_enc_dec_layers == 1:
		decoder_preds = decoder_dense(decoder_outputs1)
	elif n_enc_dec_layers == 2:
		decoder_preds = decoder_dense(decoder_outputs2)
	elif n_enc_dec_layers == 3:
		decoder_preds = decoder_dense(decoder_outputs3)

	model = keras.Model([encoder_inputs, decoder_inputs], decoder_preds)

	return model

####################### Model with Attention Layer #######################
def BuildModelAttention(Cell_Type, n_enc_dec_layers, hidden_layer_size, num_encoder_tokens, num_decoder_tokens, dropout, emb_size):
	# Encoder
	encoder_inputs = keras.Input(shape=(None,), name="enc_input")
	enc_emb =  Embedding(num_encoder_tokens, emb_size, name="enc_embedding")(encoder_inputs)

	if Cell_Type == 'RNN':
		if n_enc_dec_layers == 1:	
			encoder_outputs1, h1 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_states = [h1]
		elif n_enc_dec_layers == 2:
			encoder_outputs1, h1 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_states = [h1, h2]
		elif n_enc_dec_layers == 3:
			encoder_outputs1, h1 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_outputs3, h3 = SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer3")(encoder_outputs2)
			encoder_states = [h1, h2, h3]

	elif Cell_Type == 'LSTM':
		if n_enc_dec_layers == 1:
			encoder_outputs1, h1, c1 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_states = [h1, c1]
		elif n_enc_dec_layers == 2:
			encoder_outputs1, h1, c1 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1") (enc_emb)
			encoder_outputs2, h2, c2 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_states = [h1, c1, h2, c2]
		elif n_enc_dec_layers == 3:
			encoder_outputs1, h1, c1 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2, c2 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_outputs3, h3, c3 = LSTM(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer3")(encoder_outputs2)
			encoder_states = [h1, c1, h2, c2, h3, c3]
	elif Cell_Type == 'GRU':
		if n_enc_dec_layers == 1:
			encoder_outputs1, h1 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_states = [h1]
		elif n_enc_dec_layers == 2:
			encoder_outputs1, h1 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_states = [h1, h2]
		elif n_enc_dec_layers == 3:
			encoder_outputs1, h1 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer1")(enc_emb)
			encoder_outputs2, h2 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer2")(encoder_outputs1)
			encoder_outputs3, h3 = GRU(hidden_layer_size, return_state=True, return_sequences=True, dropout=dropout, name="enc_layer3")(encoder_outputs2)
			encoder_states = [h1, h2, h3]

	# Decoder
	decoder_inputs = keras.Input(shape=(None,), name="dec_input")
	dec_emb_layer = Embedding(num_decoder_tokens, hidden_layer_size, name="dec_embedding")
	dec_emb = dec_emb_layer(decoder_inputs)

	if Cell_Type == 'RNN':
		if n_enc_dec_layers == 1:
			decoder_outputs1, dh1 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
		elif n_enc_dec_layers == 2:
			decoder_outputs1, dh1 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
		elif n_enc_dec_layers == 3:
			decoder_outputs1, dh1 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
			decoder_outputs3, dh3 = SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer3")(decoder_outputs2, initial_state=[h3])
		
	elif Cell_Type == 'LSTM':
		if n_enc_dec_layers == 1:
			decoder_outputs1, dh1, dc1 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1, c1])
		elif n_enc_dec_layers == 2:
			decoder_outputs1, dh1, dc1 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1, c1])
			decoder_outputs2, dh2, dc2 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2, c2])
		elif n_enc_dec_layers == 3:
			decoder_outputs1, dh1, dc1 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1, c1])
			decoder_outputs2, dh2, dc2 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2, c2])
			decoder_outputs3, dh3, dc3 = LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer3")(decoder_outputs2, initial_state=[h3, c3])
	elif Cell_Type == 'GRU':
		if n_enc_dec_layers == 1:
			decoder_outputs1, dh1 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
		elif n_enc_dec_layers == 2:
			decoder_outputs1, dh1 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
		elif n_enc_dec_layers == 3:
			decoder_outputs1, dh1 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer1")(dec_emb, initial_state=[h1])
			decoder_outputs2, dh2 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer2")(decoder_outputs1, initial_state=[h2])
			decoder_outputs3, dh3 = GRU(hidden_layer_size, return_sequences=True, return_state=True, name="dec_layer3")(decoder_outputs2, initial_state=[h3])

	attn_layer = AttentionLayer(name='attention_layer')
	if n_enc_dec_layers == 1:
		attn_outputs, attn_states = attn_layer([encoder_outputs1, decoder_outputs1])
		decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs1, attn_outputs])
	elif n_enc_dec_layers == 2:
		attn_outputs, attn_states = attn_layer([encoder_outputs2, decoder_outputs2])
		decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attn_outputs])
	elif n_enc_dec_layers == 3:
		attn_outputs, attn_states = attn_layer([encoder_outputs3, decoder_outputs3])
		decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs3, attn_outputs])

	decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="dense_layer")
	dense_time = TimeDistributed(decoder_dense, name='time_distributed_layer')
	decoder_preds = dense_time(decoder_concat_input)

	model = keras.Model([encoder_inputs, decoder_inputs], decoder_preds)

	return model, decoder_dense
