import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, GRU 
from tqdm import tqdm
import pdb
import config

from LoadData import ReadData
from CreateModel import BuildModel
from Accuracy import CalculateAccuracy, BeamCalculateAccuracy
from EncoderDecoderModels import InferenceModels

WANDB = 1

if WANDB:
	import wandb
	from wandb.integration.keras import WandbMetricsLogger


	import wandb
	wandb.login(key="3d199b9bde866b3494cda2f8bb7c7a633c9fdade")
	wandb.init(project="DA6401_Assignment_3")
	wandb.init(config={"batch_size": 64, "epochs": 10, "Cell_Type": "LSTM", "h_layer_size": 64, "emb_size": 64, "dropout": 0}, project="Deep-Learning-RNN")
	myconfig = wandb.config

############################## Language ##############################
Languages = {'Bengali': 'bn', 'Gujarati': 'gu', 'Hindi': 'hi', 'Kannada': 'kn', 'Malayalam': 'ml', 'Marathi': 'mr', 'Punjabi': 'pa', 'Sindhi': 'sd', 'Sinhala': 'si', 'Tamil': 'ta', 'Telugu': 'te', 'Urdu': 'ur'}
Language = Languages['Hindi']
##################### Path to data set #####################################################

base_path = r'E:\volume d\gggg\julynov24\Deep Learning\DA6401\dakshina_dataset_v1.0\\'

train_data_path = base_path + Language + '/lexicons/' + Language + '.translit.sampled.train.tsv'
val_data_path = base_path + Language + '/lexicons/' + Language + '.translit.sampled.dev.tsv'
test_data_path = base_path + Language + '/lexicons/' + Language + '.translit.sampled.test.tsv'

#######################################################################



############ Main Program ############
def main(args):

	############################## Hyperparameters ##############################
	epochs = args.epochs                  
	optimizer = args.optimizer
	Cell_Type = args.Cell_Type
	l_rate = args.l_rate
	batch_size = args.batch_size               
	emb_size = args.embedding_size
	n_enc_dec_layers = args.n_enc_dec_layers
	hidden_layer_size = args.hidden_layer_size
	dropout = args.dropout
	beam_size = args.beam_size

	############################## Reading Train Data ##############################
	input_texts, target_texts, input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data = ReadData(train_data_path, "train")

	############################## Reading Validation Data ##############################
	val_input_texts, val_target_texts, _, _, _, _, _, _, _, _, val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = ReadData(val_data_path, "val", input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

	############################## Reading Test Data ##############################
	test_input_texts, test_target_texts, _, _, _, _, _, _, _, _, test_encoder_input_data, test_decoder_input_data, test_decoder_target_data = ReadData(test_data_path, "test", input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

	################################ Build A Model ################################
	model = BuildModel(Cell_Type, n_enc_dec_layers, hidden_layer_size, num_encoder_tokens, num_decoder_tokens, dropout, emb_size)

	################################ Train the Model ################################
	if optimizer == 'Adam':
		opt = Adam(learning_rate=l_rate, beta_1=0.9, beta_2=0.999)


	elif optimizer == 'Nadam':
		opt = Nadam(learning_rate=l_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


	model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])
	model.summary()

	### Fit the model ###
	if WANDB:
		wandb.run.name = "cell_" + Cell_Type + "_nedl_" + str(n_enc_dec_layers) + "_bms_" +  str(beam_size) + "_hls_" + str(hidden_layer_size) + "_embs_" + str(emb_size) + "_ep_" + str(epochs) + "_bs_" + str(batch_size) + "_op_" + optimizer + "_do_" + str(dropout) + "_lr_" + str(l_rate)
		model.fit(
		    [encoder_input_data, decoder_input_data],
		    decoder_target_data,
		    batch_size=batch_size,
		    epochs=epochs,
		    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
		    callbacks=[WandbMetricsLogger()]


		)
	else:
		model.fit(
	    		[encoder_input_data, decoder_input_data],
	    		decoder_target_data,
	    		batch_size=batch_size,
	    		epochs=epochs,
	    		validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data)
			)

	################################ Save Model ################################
	model.save("s2s.keras")     


	################################ Inference Models ################################
	encoder_model, decoder_model = InferenceModels(model, Cell_Type, hidden_layer_size, n_enc_dec_layers)

	# Reverse-lookup token index to decode sequences back to something readable.
	reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
	reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

	# Count
	n_val_words = len(val_input_texts)
	n_train_words = len(input_texts)
	n_test_words = len(test_input_texts)
	
	################################ Calculate Accuracy ################################
	print('\n CALCULATING WORD-LEVEL ACCURACY!! \n')

	# Beam Search
	if beam_size > 1:
		print("Train Data")
		train_acc = BeamCalculateAccuracy(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, n_train_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
		print("Validation Data")
		val_acc = BeamCalculateAccuracy(val_encoder_input_data, encoder_model, decoder_model, val_input_texts, val_target_texts, n_val_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
		print("Test Data")
		test_acc = BeamCalculateAccuracy(test_encoder_input_data, encoder_model, decoder_model, test_input_texts, test_target_texts, n_test_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers, beam_size)
	# No Beam Search
	else:
		print("Train Data")
		train_acc = CalculateAccuracy(encoder_input_data, encoder_model, decoder_model, input_texts, target_texts, n_train_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)
		print("Validation Data")
		val_acc = CalculateAccuracy(val_encoder_input_data, encoder_model, decoder_model, val_input_texts, val_target_texts, n_val_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)
		print("Test Data")
		test_acc = CalculateAccuracy(test_encoder_input_data, encoder_model, decoder_model, test_input_texts, test_target_texts, n_test_words, max_decoder_seq_length, target_token_index, reverse_target_char_index, Cell_Type, n_enc_dec_layers)

	if WANDB:
		wandb.log({"word_level_acc": acc})
	print("Train Accuracy (exact string match): %f " % (train_acc))
	print("Validation Accuracy (exact string match): %f " % (val_acc))
	print("Test Accuracy (exact string match): %f " % (test_acc))



############################ Main Funtion ############################
if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
