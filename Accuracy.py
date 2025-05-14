from DecodeText import DecodeSequence, DecodeSequenceAttention, BeamDecodeSequence, BeamDecodeSequenceAttention
from tqdm import tqdm
import pdb

######### Accuracy #########
def CalculateAccuracy(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, total_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers):

	count = 0

	for seq_index in tqdm(range(total_words), desc='Transliteration in Progress'):

		input_seq = encoder_input_data[seq_index : seq_index + 1]

		decoded_word = DecodeSequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)
		'''		
		print("-")
		print("Input Word:", input_texts[seq_index])
		print("Actual Transliterated Word", target_texts[seq_index][1:-1])
		print("Decoded Transliterated Word:", decoded_word)
		pdb.set_trace()
		'''

		if(target_texts[seq_index][1:-1] == decoded_word):
			count = count + 1

	accuracy = count/total_words
	return accuracy

######### Accuracy BeamSearch #########
def BeamCalculateAccuracy(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, total_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size):

	count = 0
	for seq_index in tqdm(range(total_words), desc='Transliteration in Progress'):
		
		input_seq = encoder_input_data[seq_index : seq_index + 1]
		suggestions = BeamDecodeSequence(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
		
		if target_texts[seq_index][1:-1] in suggestions:
			count = count + 1

	accuracy = count/total_words
	return accuracy

######### Accuracy Attention #########
def CalculateAccuracyAttention(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, total_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers):

	count = 0
	for seq_index in tqdm(range(total_words), desc='Transliteration in Progress'):

		input_seq = encoder_input_data[seq_index : seq_index + 1]

		decoded_word, attention_weights = DecodeSequenceAttention(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)

		if(target_texts[seq_index][1:-1] == decoded_word):
			count = count + 1

	accuracy = count/total_words
	return accuracy

######### Accuracy Attention+BeamSearch #########
def BeamCalculateAccuracyAttention(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, total_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size):

	count = 0
	for seq_index in tqdm(range(total_words), desc='Transliteration in Progress'):
		
		input_seq = encoder_input_data[seq_index : seq_index + 1]
		suggestions, attention_weights = BeamDecodeSequenceAttention(input_seq, encoder_model, decoder_model, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
		
		if target_texts[seq_index][1:-1] in suggestions:
			count = count + 1

	accuracy = count/total_words
	return accuracy
