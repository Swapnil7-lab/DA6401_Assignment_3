from tensorflow import keras
import pdb

def InferenceModels(model, Cell_Type, hidden_layer_size, n_enc_dec_layers):
	
	if n_enc_dec_layers == 1:
		encoder_inputs = model.input[0]

		if Cell_Type == 'RNN':
			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_states = [h1]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h]

			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_rnn = model.get_layer('dec_layer1')

			decoder_outputs1, dh1 = decoder_rnn(dec_emb, initial_state=decoder_states_inputs)
			decoder_states = [dh1]

		elif Cell_Type == 'LSTM':
			encoder_outputs1, h1, c1 = model.get_layer('enc_layer1').output
			encoder_states = [h1, c1]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_lstm = model.get_layer('dec_layer1')
			decoder_outputs1, dh1, dc1 = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
			decoder_states = [dh1, dc1]

		elif Cell_Type == 'GRU':
			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_states = [h1]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h]
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_gru = model.get_layer('dec_layer1')
			decoder_outputs1, dh1 = decoder_gru(dec_emb, initial_state=decoder_states_inputs)
			decoder_states = [dh1]

		decoder_dense = model.get_layer('dense_layer')
		decoder_preds = decoder_dense(decoder_outputs1)

	if n_enc_dec_layers == 2:
		encoder_inputs = model.input[0]

		if Cell_Type == 'RNN':
			
			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
	
			encoder_states = [h1, h2]
			encoder_model = keras.Model(encoder_inputs, encoder_states)
			decoder_inputs = model.input[1]
			
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_rnn_layer1 = model.get_layer('dec_layer1')
			decoder_rnn_layer2 = model.get_layer('dec_layer2')

			decoder_outputs1, dh1 = decoder_rnn_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_rnn_layer2(decoder_outputs1, initial_state=decoder_states_inputs[-1:])
			decoder_states = [dh1, dh2]

		elif Cell_Type == 'LSTM':

			encoder_outputs1, h1, c1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2, c2 = model.get_layer('enc_layer2').output
			
			encoder_states = [h1, c1, h2, c2]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c1 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_h1, decoder_state_input_c1]

			dec_emb_layer = model.get_layer('dec_embedding') 
			dec_emb = dec_emb_layer(decoder_inputs)
			decoder_lstm_layer1 = model.get_layer('dec_layer1')
			decoder_lstm_layer2 = model.get_layer('dec_layer2')

			decoder_outputs1, dh1, dc1 = decoder_lstm_layer1(dec_emb, initial_state=decoder_states_inputs[:2])
			decoder_outputs2, dh2, dc2 = decoder_lstm_layer2(decoder_outputs1, initial_state=decoder_states_inputs[-2:])
			decoder_states = [dh1, dc1, dh2, dc2]
			
		elif Cell_Type == 'GRU':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output

			encoder_states = [h1, h2]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_gru_layer1 = model.get_layer('dec_layer1')
			decoder_gru_layer2 = model.get_layer('dec_layer2')

			decoder_outputs1, dh1 = decoder_gru_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_gru_layer2(decoder_outputs1, initial_state=decoder_states_inputs[-1:])
			decoder_states = [dh1, dh2]

		decoder_dense = model.get_layer('dense_layer')
		decoder_preds = decoder_dense(decoder_outputs2)

	if n_enc_dec_layers == 3:
		encoder_inputs = model.input[0]

		if Cell_Type == 'RNN':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
			encoder_outputs3, h3 = model.get_layer('enc_layer3').output
			
			encoder_states = [h1, h2, h3]
			encoder_model = keras.Model(encoder_inputs, encoder_states)
			decoder_inputs = model.input[1]
			
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h2 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1, decoder_state_input_h2]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_rnn_layer1 = model.get_layer('dec_layer1')
			decoder_rnn_layer2 = model.get_layer('dec_layer2')
			decoder_rnn_layer3 = model.get_layer('dec_layer3')

			decoder_outputs1, dh1 = decoder_rnn_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_rnn_layer2(decoder_outputs1, initial_state=decoder_states_inputs[1])
			decoder_outputs3, dh3 = decoder_rnn_layer3(decoder_outputs2, initial_state=decoder_states_inputs[2])
			decoder_states = [dh1, dh2, dh3]

		elif Cell_Type == 'LSTM':
			
			encoder_outputs1, h1, c1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2, c2 = model.get_layer('enc_layer2').output
			encoder_outputs3, h3, c3 = model.get_layer('enc_layer3').output

			encoder_states = [h1, c1, h2, c2, h3, c3]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h2 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c2 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2]

			dec_emb_layer = model.get_layer('dec_embedding') 
			dec_emb = dec_emb_layer(decoder_inputs)
			decoder_lstm_layer1 = model.get_layer('dec_layer1')
			decoder_lstm_layer2 = model.get_layer('dec_layer2')
			decoder_lstm_layer3 = model.get_layer('dec_layer3')
			decoder_outputs1, dh1, dc1 = decoder_lstm_layer1(dec_emb, initial_state=decoder_states_inputs[:2])
			decoder_outputs2, dh2, dc2 = decoder_lstm_layer2(decoder_outputs1, initial_state=decoder_states_inputs[2:4])
			decoder_outputs3, dh3, dc3 = decoder_lstm_layer3(decoder_outputs2, initial_state=decoder_states_inputs[4:6])
			decoder_states = [dh1, dc1, dh2, dc2, dh3, dc3]
			
		elif Cell_Type == 'GRU':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
			encoder_outputs3, h3 = model.get_layer('enc_layer3').output

			encoder_states = [h1, h2, h3]
			encoder_model = keras.Model(encoder_inputs, encoder_states)

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h2 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1, decoder_state_input_h2]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_gru_layer1 = model.get_layer('dec_layer1')
			decoder_gru_layer2 = model.get_layer('dec_layer2')
			decoder_gru_layer3 = model.get_layer('dec_layer3')

			decoder_outputs1, dh1  = decoder_gru_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_gru_layer2(decoder_outputs1, initial_state=decoder_states_inputs[1])
			decoder_outputs3, dh3 = decoder_gru_layer3(decoder_outputs2, initial_state=decoder_states_inputs[2])
			decoder_states = [dh1, dh2, dh3]

		decoder_dense = model.get_layer('dense_layer')
		decoder_preds = decoder_dense(decoder_outputs3)

	decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_preds] + decoder_states)

	return encoder_model, decoder_model

############# Inference Models for Attention #############
def InferenceModelsAttention(model, Cell_Type, hidden_layer_size, n_enc_dec_layers, decoder_dense):
	
	if n_enc_dec_layers == 1:
		encoder_inputs = model.input[0]

		if Cell_Type == 'RNN':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_states = [h1]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs1, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h]

			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_rnn = model.get_layer('dec_layer1')
			decoder_outputs1, dh1 = decoder_rnn(dec_emb, initial_state=decoder_states_inputs)
			decoder_states = [dh1]

		elif Cell_Type == 'LSTM':

			encoder_outputs1, h1, c1 = model.get_layer('enc_layer1').output
			encoder_states = [h1, c1]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs1, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_lstm = model.get_layer('dec_layer1')
			decoder_outputs1, dh1, dc1 = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
			decoder_states = [dh1, dc1]

		elif Cell_Type == 'GRU':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_states = [h1]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs1, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h]

			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_gru = model.get_layer('dec_layer1')
			decoder_outputs1, dh1 = decoder_gru(dec_emb, initial_state=decoder_states_inputs)
			decoder_states = [dh1]

		attn_layer = model.get_layer('attention_layer')
		attn_outputs, attn_states = attn_layer([encoder_state_input, decoder_outputs1])
		decoder_concat_input = model.get_layer('concat_layer')([decoder_outputs1, attn_outputs])

	if n_enc_dec_layers == 2:
		encoder_inputs = model.input[0]

		if Cell_Type == 'RNN':
			
			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
			encoder_states = [h1, h2]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs2, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_rnn_layer1 = model.get_layer('dec_layer1')
			decoder_rnn_layer2 = model.get_layer('dec_layer2')

			decoder_outputs1, dh1 = decoder_rnn_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_rnn_layer2(decoder_outputs1, initial_state=decoder_states_inputs[-1:])
			decoder_states = [dh1, dh2]

		elif Cell_Type == 'LSTM':

			encoder_outputs1, h1, c1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2, c2 = model.get_layer('enc_layer2').output
			encoder_states = [h1, c1, h2, c2]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs2, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]			
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c1 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_h1, decoder_state_input_c1]

			dec_emb_layer = model.get_layer('dec_embedding') 
			dec_emb = dec_emb_layer(decoder_inputs)
			decoder_lstm_layer1 = model.get_layer('dec_layer1')
			decoder_lstm_layer2 = model.get_layer('dec_layer2')

			decoder_outputs1, dh1, dc1 = decoder_lstm_layer1(dec_emb, initial_state=decoder_states_inputs[:2])
			decoder_outputs2, dh2, dc2 = decoder_lstm_layer2(decoder_outputs1, initial_state=decoder_states_inputs[-2:])
			decoder_states = [dh1, dc1, dh2, dc2]
			
		elif Cell_Type == 'GRU':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
			encoder_states = [h1, h2]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs2, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_gru_layer1 = model.get_layer('dec_layer1')
			decoder_gru_layer2 = model.get_layer('dec_layer2')

			decoder_outputs1, dh1 = decoder_gru_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_gru_layer2(decoder_outputs1, initial_state=decoder_states_inputs[-1:])
			decoder_states = [dh1, dh2]

		attn_layer = model.get_layer('attention_layer')
		attn_outputs, attn_states = attn_layer([encoder_state_input, decoder_outputs2])
		decoder_concat_input = model.get_layer('concat_layer')([decoder_outputs2, attn_outputs])

	if n_enc_dec_layers == 3:
		encoder_inputs = model.input[0]

		if Cell_Type == 'RNN':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
			encoder_outputs3, h3 = model.get_layer('enc_layer3').output
			encoder_states = [h1, h2, h3]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs3, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]	
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h2 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1, decoder_state_input_h2]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_rnn_layer1 = model.get_layer('dec_layer1')
			decoder_rnn_layer2 = model.get_layer('dec_layer2')
			decoder_rnn_layer3 = model.get_layer('dec_layer3')

			decoder_outputs1, dh1 = decoder_rnn_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_rnn_layer2(decoder_outputs1, initial_state=decoder_states_inputs[1])
			decoder_outputs3, dh3 = decoder_rnn_layer3(decoder_outputs2, initial_state=decoder_states_inputs[2])
			decoder_states = [dh1, dh2, dh3]

		elif Cell_Type == 'LSTM':
			
			encoder_outputs1, h1, c1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2, c2 = model.get_layer('enc_layer2').output
			encoder_outputs3, h3, c3 = model.get_layer('enc_layer3').output
			encoder_states = [h1, c1, h2, c2, h3, c3]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs3, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h2 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_c2 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2]

			dec_emb_layer = model.get_layer('dec_embedding') 
			dec_emb = dec_emb_layer(decoder_inputs)
			decoder_lstm_layer1 = model.get_layer('dec_layer1')
			decoder_lstm_layer2 = model.get_layer('dec_layer2')
			decoder_lstm_layer3 = model.get_layer('dec_layer3')
			decoder_outputs1, dh1, dc1 = decoder_lstm_layer1(dec_emb, initial_state=decoder_states_inputs[:2])
			decoder_outputs2, dh2, dc2 = decoder_lstm_layer2(decoder_outputs1, initial_state=decoder_states_inputs[2:4])
			decoder_outputs3, dh3, dc3 = decoder_lstm_layer3(decoder_outputs2, initial_state=decoder_states_inputs[4:6])
			decoder_states = [dh1, dc1, dh2, dc2, dh3, dc3]
			
		elif Cell_Type == 'GRU':

			encoder_outputs1, h1 = model.get_layer('enc_layer1').output
			encoder_outputs2, h2 = model.get_layer('enc_layer2').output
			encoder_outputs3, h3 = model.get_layer('enc_layer3').output
			encoder_states = [h1, h2, h3]
			encoder_model = keras.Model(encoder_inputs, [encoder_outputs3, encoder_states])

			encoder_state_input = keras.Input(shape=(None, hidden_layer_size,))

			decoder_inputs = model.input[1]
			decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
			decoder_state_input_h2 = keras.Input(shape=(hidden_layer_size,))
			decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1, decoder_state_input_h2]
			
			dec_emb_layer = model.get_layer('dec_embedding')
			dec_emb = dec_emb_layer(decoder_inputs)

			decoder_gru_layer1 = model.get_layer('dec_layer1')
			decoder_gru_layer2 = model.get_layer('dec_layer2')
			decoder_gru_layer3 = model.get_layer('dec_layer3')

			decoder_outputs1, dh1  = decoder_gru_layer1(dec_emb, initial_state=decoder_states_inputs[:1])
			decoder_outputs2, dh2 = decoder_gru_layer2(decoder_outputs1, initial_state=decoder_states_inputs[1])
			decoder_outputs3, dh3 = decoder_gru_layer3(decoder_outputs2, initial_state=decoder_states_inputs[2])
			decoder_states = [dh1, dh2, dh3]

		attn_layer = model.get_layer('attention_layer')
		attn_outputs, attn_states = attn_layer([encoder_state_input, decoder_outputs3])
		decoder_concat_input = model.get_layer('concat_layer')([decoder_outputs3, attn_outputs])

	decoder_preds = keras.layers.TimeDistributed(decoder_dense)(decoder_concat_input)
	decoder_model = keras.Model(inputs=[encoder_state_input] + decoder_states_inputs + [decoder_inputs], outputs=[decoder_preds] + [attn_states] + decoder_states)

	return encoder_model, decoder_model
