import re
# Imports for visualisations
from IPython.display import HTML as html_print
from IPython.display import display
import keras.backend as K

# get html element
def cstr(s, color='black'):
	if s == ' ':
		return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
	else:
		return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)
	
# print html
def print_color(t):
	display(html_print(''.join([cstr(ti, color=ci) for ti,ci in t])))

# get appropriate color for value
def get_clr(value):
	colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
	value = int((value * 100) / 5)
	if value > len(colors)-1:
		value = len(colors)-1
	return colors[value]

# sigmoid function
def sigmoid(x):
	z = 1/(1 + np.exp(-x)) 
	return z

# To visualize connectivity
def visualize(attn_mat, result_list, input_timestep, input_list):
	print(input_list[input_timestep])
 	text_colours = []
	for output_timestep in range(len(result_list)):
		text = (result_list[output_timestep], get_clr(attn_mat[output_timestep][input_timestep]))
		text_colours.append(text)
	print_color(text_colours)
