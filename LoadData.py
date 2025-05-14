import numpy as np
import csv
import pdb
from sklearn.utils import shuffle

############################## Reading Data ##############################
def ReadData(data_path, category, inp_chars=None, tar_chars=None, max_enc_seq_len=None, max_dec_seq_len=None, num_enc_tok=None, num_dec_tok=None, inp_tok_index=None, tar_tok_index=None):
	input_texts = []
	target_texts = []
	input_characters = set()
	target_characters = set()

	with open(data_path, encoding='utf-8') as tsv_file:

	
		tsvreader = csv.reader(tsv_file, delimiter="\t")
		for line in tsvreader:
			input_text = line[1:2]
			target_text = line[0:1]

			input_text = ' '.join([str(elem) for elem in input_text])
			target_text = ' '.join([str(elem) for elem in target_text])

			target_text = "#" + target_text + "$"

			input_texts.append(input_text)
			target_texts.append(target_text)
		
			for char in input_text:
				if char not in input_characters:
					input_characters.add(char)
			for char in target_text:
				if char not in target_characters:
					target_characters.add(char)	

	############ Appending Spaces ############
	input_texts.append(" ")
	target_texts.append(" ")
	# Shuffle the data
	#input_texts, target_texts = shuffle(input_texts, target_texts, random_state=0)
	input_characters.add(" ")
	target_characters.add(" ")

	if category == "train":
		input_characters = sorted(list(input_characters))
		target_characters = sorted(list(target_characters))
		num_encoder_tokens = len(input_characters)
		num_decoder_tokens = len(target_characters)
		max_encoder_seq_length = max([len(txt) for txt in input_texts])
		max_decoder_seq_length = max([len(txt) for txt in target_texts])
	elif category == "val" or category == "test":
		input_characters = inp_chars
		target_characters = tar_chars
		num_encoder_tokens = num_enc_tok
		num_decoder_tokens = num_dec_tok
		max_encoder_seq_length = max_enc_seq_len
		max_decoder_seq_length = max_dec_seq_len
		
		
	print("Number of samples:", len(input_texts))
	print("Number of unique input tokens:", num_encoder_tokens)
	print("Number of unique output tokens:", num_decoder_tokens)
	print("Max sequence length for inputs:", max_encoder_seq_length)
	print("Max sequence length for outputs:", max_decoder_seq_length)

	input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
	target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

	encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")
	decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
	decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

	for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
	
		for t, char in enumerate(input_text):
			encoder_input_data[i, t] = input_token_index[char]

		encoder_input_data[i, t + 1 :] = input_token_index[" "]

		for t, char in enumerate(target_text):
			decoder_input_data[i, t] = target_token_index[char]
			if t > 0:
				decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

		decoder_input_data[i, t + 1 :] = target_token_index[" "]
		decoder_target_data[i, t:, target_token_index[" "]] = 1.0

	tsv_file. close()

	return input_texts, target_texts, input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data
