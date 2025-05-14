import numpy as np
import pdb
import math


def DecodeSequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_layers=1):

	# Encode the input as state vectors.
	if Cell_Type == 'RNN':
		if n_layers == 1:
			states_value = [encoder_model.predict(input_seq)]
		else:
			states_value = encoder_model.predict(input_seq)
	elif Cell_Type == 'LSTM':
		states_value = encoder_model.predict(input_seq)
	elif Cell_Type == 'GRU':
		if n_layers == 1:
			states_value = [encoder_model.predict(input_seq)]
		else:
			states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1,1))

	# Populate the first character of target sequence with the start character.
	target_seq[0, 0] = target_token_index['#']

	# Sampling loop
	stop_condition = False
	decoded_sentence = ''

	while not stop_condition:
		# Decode
		if Cell_Type == 'RNN':
			if n_layers == 1:
				output_tokens, h = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 2:
				output_tokens, h, h1 = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 3:
				output_tokens, h, h1, h2 = decoder_model.predict([target_seq] + states_value)
		elif Cell_Type == 'LSTM':
			if n_layers == 1:
				output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 2:
				output_tokens, h, c, h1, c1 = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 3:
				output_tokens, h, c, h1, c1, h2, c2 = decoder_model.predict([target_seq] + states_value)
		elif Cell_Type == 'GRU':
			if n_layers == 1:
				output_tokens, h = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 2:
				output_tokens, h, h1 = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 3:
				output_tokens, h, h1, h2 = decoder_model.predict([target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]

		# Exit condition: either hit max length or find stop character.
		if sampled_char == "$" or len(decoded_sentence) > max_decoder_seq_length:
			stop_condition = True
		else:
			decoded_sentence += sampled_char

			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1))
			target_seq[0, 0] = sampled_token_index

			# Update states
			if Cell_Type == 'RNN':
				if n_layers == 1:
					states_value = [h]
				elif n_layers == 2:
					states_value = [h, h1]
				elif n_layers == 3:
					states_value = [h, h1, h2]
			elif Cell_Type == 'LSTM':
				if n_layers == 1:
					states_value = [h, c]
				elif n_layers == 2:
					states_value = [h, c, h1, c1]
				elif n_layers == 3:
					states_value = [h, c, h1, c1, h2, c2]
			elif Cell_Type == 'GRU':
				if n_layers == 1:
					states_value = [h]
				elif n_layers == 2:
					states_value = [h, h1]
				elif n_layers == 3:
					states_value = [h, h1, h2]

	return decoded_sentence

def beam_search_decoder(predictions, top_k):
	#start with an empty sequence with zero score
	output_sequences = [([], 0)]
	#looping through all the predictions
	for token_probs in predictions:
		new_sequences = []
		#append new tokens to old sequences and re-score
		for old_seq, old_score in output_sequences:
			for char_index in range(len(token_probs)):
				new_seq = old_seq + [char_index]
			       		
				#considering log-likelihood for scoring
				likelihood = token_probs[char_index]
				
				new_score = old_score + math.log(likelihood)
				new_sequences.append((new_seq, new_score))
		        
		#sort all new sequences in the de-creasing order of their score
		output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        
		#select top-k based on score  Note- best sequence is with the highest score
		output_sequences = output_sequences[:top_k]
        
	return output_sequences

def BeamDecodeSequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_layers, beam_size):
	
	# Encode the input as state vectors.
	if Cell_Type == 'RNN':
		if n_layers == 1:
			states_value = [encoder_model.predict(input_seq)]
		else:
			states_value = encoder_model.predict(input_seq)
	elif Cell_Type == 'LSTM':
		states_value = encoder_model.predict(input_seq)
	elif Cell_Type == 'GRU':
		if n_layers == 1:
			states_value = [encoder_model.predict(input_seq)]
		else:
			states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1,1))

	# Populate the first character of target sequence with the start character.
	target_seq[0, 0] = target_token_index['#']

	# Sampling loop
	stop_condition = False
	decoded_sentence = ''
	output_prob = []
	
	while not stop_condition:
		# Decode
		if Cell_Type == 'RNN':
			if n_layers == 1:
				output_tokens, h = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 2:
				output_tokens, h, h1 = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 3:
				output_tokens, h, h1, h2 = decoder_model.predict([target_seq] + states_value)
		elif Cell_Type == 'LSTM':
			if n_layers == 1:
				output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 2:
				output_tokens, h, c, h1, c1 = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 3:
				output_tokens, h, c, h1, c1, h2, c2 = decoder_model.predict([target_seq] + states_value)
		elif Cell_Type == 'GRU':
			if n_layers == 1:
				output_tokens, h = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 2:
				output_tokens, h, h1 = decoder_model.predict([target_seq] + states_value)
			elif n_layers == 3:
				output_tokens, h, h1, h2 = decoder_model.predict([target_seq] + states_value)
			
		
		output_prob.append(output_tokens.tolist())         		
		
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]

		# Exit condition: either hit max length or find stop character.
		if sampled_char == "$" or len(decoded_sentence) > max_decoder_seq_length:
			stop_condition = True
			output_prob = sum(output_prob, [])
			output = [elem for twod in output_prob for elem in twod]
			sequences = beam_search_decoder(output, beam_size)	
		else:
			decoded_sentence += sampled_char

			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1))
			target_seq[0, 0] = sampled_token_index

			# Update states
			if Cell_Type == 'RNN':
				if n_layers == 1:
					states_value = [h]
				elif n_layers == 2:
					states_value = [h, h1]
				elif n_layers == 3:
					states_value = [h, h1, h2]
			elif Cell_Type == 'LSTM':
				if n_layers == 1:
					states_value = [h, c]
				elif n_layers == 2:
					states_value = [h, c, h1, c1]
				elif n_layers == 3:
					states_value = [h, c, h1, c1, h2, c2]
			elif Cell_Type == 'GRU':
				if n_layers == 1:
					states_value = [h]
				elif n_layers == 2:
					states_value = [h, h1]
				elif n_layers == 3:
					states_value = [h, h1, h2]

	suggestions = list()
	for k in range(beam_size):
		decoded_word = ''		
		encoded_list = sequences[k][0]
		for i in range(len(encoded_list)):
			sampled_token_index = sequences[k][0][i]
			sampled_char = reverse_target_char_index[sampled_token_index]
			decoded_word += sampled_char					
	
		suggestions.append(decoded_word[:-1])	
		
	return suggestions


############ Attention Decode Sequence ############
def DecodeSequenceAttention(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_layers=1):

	# Encode the input as state vectors.
	if Cell_Type == 'RNN':
		if n_layers == 1:
			encoder_outputs, h1 = encoder_model.predict(input_seq)
			states_value = [h1]
		elif n_layers == 2:
			encoder_outputs, h1, h2 = encoder_model.predict(input_seq)
			states_value = [h1, h2]
		elif n_layers == 3:
			encoder_outputs, h1, h2, h3 = encoder_model.predict(input_seq)
			states_value = [h1, h2, h3]
	elif Cell_Type == 'LSTM':
		if n_layers == 1:
			encoder_outputs, h1, c1 = encoder_model.predict(input_seq)
			states_value = [h1, c1]
		elif n_layers == 2:
			encoder_outputs, h1, c1, h2, c2 = encoder_model.predict(input_seq)
			states_value = [h1, c1, h2, c2]
		elif n_layers == 3:
			encoder_outputs, h1, c1, h2, c2, h3, c3 = encoder_model.predict(input_seq)
			states_value = [h1, c1, h2, c2, h3, c3]
	elif Cell_Type == 'GRU':
		if n_layers == 1:
			encoder_outputs, h1 = encoder_model.predict(input_seq)
			states_value = [h1]
		elif n_layers == 2:
			encoder_outputs, h1, h2 = encoder_model.predict(input_seq)
			states_value = [h1, h2]
		elif n_layers == 3:
			encoder_outputs, h1, h2, h3 = encoder_model.predict(input_seq)
			states_value = [h1, h2, h3]

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1,1))

	# Populate the first character of target sequence with the start character.
	target_seq[0, 0] = target_token_index['#']
	attention_weights = []

	# Sampling loop
	stop_condition = False
	decoded_sentence = ''

	while not stop_condition:
		# Decode
		if Cell_Type == 'RNN':
			if n_layers == 1:
				output_tokens, attention, dh1 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 2:
				output_tokens, attention, dh1, dh2 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 3:
				output_tokens, attention, dh1, dh2, dh3 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
		elif Cell_Type == 'LSTM':
			if n_layers == 1:
				output_tokens, attention, dh1, dc1 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 2:
				output_tokens, attention, dh1, dc1, dh2, dc2 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 3:
				output_tokens, attention, dh1, dc1, dh2, dc2, dh3, dc3 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
		elif Cell_Type == 'GRU':
			if n_layers == 1:
				output_tokens, attention, dh1 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 2:
				output_tokens, attention, dh1, dh2 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 3:
				output_tokens, attention, dh1, dh2, dh3 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]

		# Exit condition: either hit max length or find stop character.
		if sampled_char == "$" or len(decoded_sentence) > max_decoder_seq_length:
			stop_condition = True
		else:
			attention_weights.append((sampled_token_index, attention))
			decoded_sentence += sampled_char

			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1))
			target_seq[0, 0] = sampled_token_index

			# Update states
			if Cell_Type == 'RNN':
				if n_layers == 1:
					states_value = [dh1]
				elif n_layers == 2:
					states_value = [dh1, dh2]
				elif n_layers == 3:
					states_value = [dh1, dh2, dh3]
			elif Cell_Type == 'LSTM':
				if n_layers == 1:
					states_value = [dh1, dc1]
				elif n_layers == 2:
					states_value = [dh1, dc1, dh2, dc2]
				elif n_layers == 3:
					states_value = [dh1, dc1, dh2, dc2, dh3, dc3]
			elif Cell_Type == 'GRU':
				if n_layers == 1:
					states_value = [dh1]
				elif n_layers == 2:
					states_value = [dh1, dh2]
				elif n_layers == 3:
					states_value = [dh1, dh2, dh3]

	return decoded_sentence, attention_weights


############ Attention+BeamSearch Decode Sequence ############
def BeamDecodeSequenceAttention(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_layers, beam_size):
	
	# Encode the input as state vectors.
	if Cell_Type == 'RNN':
		if n_layers == 1:
			encoder_outputs, h1 = encoder_model.predict(input_seq)
			states_value = [h1]
		elif n_layers == 2:
			encoder_outputs, h1, h2 = encoder_model.predict(input_seq)
			states_value = [h1, h2]
		elif n_layers == 3:
			encoder_outputs, h1, h2, h3 = encoder_model.predict(input_seq)
			states_value = [h1, h2, h3]
	elif Cell_Type == 'LSTM':
		if n_layers == 1:
			encoder_outputs, h1, c1 = encoder_model.predict(input_seq)
			states_value = [h1, c1]
		elif n_layers == 2:
			encoder_outputs, h1, c1, h2, c2 = encoder_model.predict(input_seq)
			states_value = [h1, c1, h2, c2]
		elif n_layers == 3:
			encoder_outputs, h1, c1, h2, c2, h3, c3 = encoder_model.predict(input_seq)
			states_value = [h1, c1, h2, c2, h3, c3]
	elif Cell_Type == 'GRU':
		if n_layers == 1:
			encoder_outputs, h1 = encoder_model.predict(input_seq)
			states_value = [h1]
		elif n_layers == 2:
			encoder_outputs, h1, h2 = encoder_model.predict(input_seq)
			states_value = [h1, h2]
		elif n_layers == 3:
			encoder_outputs, h1, h2, h3 = encoder_model.predict(input_seq)
			states_value = [h1, h2, h3]

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1,1))

	# Populate the first character of target sequence with the start character.
	target_seq[0, 0] = target_token_index['#']
	attention_weights = []

	# Sampling loop
	stop_condition = False
	decoded_sentence = ''
	output_prob = []
	
	while not stop_condition:
		# Decode
		if Cell_Type == 'RNN':
			if n_layers == 1:
				output_tokens, attention, dh1 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 2:
				output_tokens, attention, dh1, dh2 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 3:
				output_tokens, attention, dh1, dh2, dh3 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
		elif Cell_Type == 'LSTM':
			if n_layers == 1:
				output_tokens, attention, dh1, dc1 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 2:
				output_tokens, attention, dh1, dc1, dh2, dc2 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 3:
				output_tokens, attention, dh1, dc1, dh2, dc2, dh3, dc3 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
		elif Cell_Type == 'GRU':
			if n_layers == 1:
				output_tokens, attention, dh1 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 2:
				output_tokens, attention, dh1, dh2 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			elif n_layers == 3:
				output_tokens, attention, dh1, dh2, dh3 = decoder_model.predict([encoder_outputs] + states_value + [target_seq])
			
		
		output_prob.append(output_tokens.tolist())         		
		
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]

		# Exit condition: either hit max length or find stop character.
		if sampled_char == "$" or len(decoded_sentence) > max_decoder_seq_length:
			stop_condition = True
			output_prob = sum(output_prob, [])
			output = [elem for twod in output_prob for elem in twod]
			sequences = beam_search_decoder(output, beam_size)	
		else:
			attention_weights.append((sampled_token_index, attention))
			decoded_sentence += sampled_char

			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1))
			target_seq[0, 0] = sampled_token_index

			# Update states
			if Cell_Type == 'RNN':
				if n_layers == 1:
					states_value = [dh1]
				elif n_layers == 2:
					states_value = [dh1, dh2]
				elif n_layers == 3:
					states_value = [dh1, dh2, dh3]
			elif Cell_Type == 'LSTM':
				if n_layers == 1:
					states_value = [dh1, dc1]
				elif n_layers == 2:
					states_value = [dh1, dc1, dh2, dc2]
				elif n_layers == 3:
					states_value = [dh1, dc1, dh2, dc2, dh3, dc3]
			elif Cell_Type == 'GRU':
				if n_layers == 1:
					states_value = [dh1]
				elif n_layers == 2:
					states_value = [dh1, dh2]
				elif n_layers == 3:
					states_value = [dh1, dh2, dh3]

	suggestions = list()
	for k in range(beam_size):
		decoded_word = ''		
		encoded_list = sequences[k][0]
		for i in range(len(encoded_list)):
			sampled_token_index = sequences[k][0][i]
			sampled_char = reverse_target_char_index[sampled_token_index]
			decoded_word += sampled_char					
	
		suggestions.append(decoded_word[:-1])	
		
	return suggestions, attention_weights
