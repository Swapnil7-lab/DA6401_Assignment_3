import argparse

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--optimizer', type=str, default="Adam")
	parser.add_argument('--Cell_Type', type=str, default="GRU")
	parser.add_argument('--l_rate', type=float, default=0.001)
	parser.add_argument('--loss', type=str, default='categorical_crossentropy')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--embedding_size', type=int, default=64)
	parser.add_argument('--hidden_layer_size', type=int, default=64)
	parser.add_argument('--n_enc_dec_layers', type=int, default=1)
	parser.add_argument('--beam_size', type=int, default=1)
	parser.add_argument('--dropout', type=float, default=0.0)
	args = parser.parse_args()

	return args
