import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb

def PlotAttentionWeights(encoder_inputs, len_input, attention_weights, en_id2char, hin_id2char, filename=None):
    """
    Plots attention weights
    :param encoder_inputs: Sequence of word ids (list/numpy.ndarray)
    :param attention_weights: Sequence of (<word_id_at_decode_step_t>:<attention_weights_at_decode_step_t>)
    :param en_id2char: dict
    :param hin_id2char: dict
    :return:
    """
    mpl.rcParams['font.sans-serif'] = ['Source Han Sans TW', 'sans-serif', "Lohit Devanagari"]

    if len(attention_weights) == 0:
        print('Your attention weights was empty. No attention map saved to the disk. ' +
              '\nPlease check if the decoder produced  a proper translation')
        return

    mats = []
    pred_char_seq = []
    for dec_ind, attn in attention_weights:
        mats.append(attn.reshape(-1)[0:len_input])
        pred_char_seq.append(dec_ind)
    #attention_mat = np.transpose(np.array(mats))
    attention_mat = np.array(mats)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(attention_mat)

    ax.set_xticks(np.arange(attention_mat.shape[1]))
    ax.set_yticks(np.arange(attention_mat.shape[0]))

    xt = [en_id2char[inp] if inp != 0 else " " for inp in encoder_inputs.ravel()[0:len_input]]
    yt = [hin_id2char[inp] if inp != 0 else " " for inp in pred_char_seq]
    ax.set_xticklabels(xt)
    ax.set_yticklabels(yt)

    ax.tick_params(labelsize=40)
    ax.tick_params(axis='x', labelrotation=0)

    plt.savefig(r'E:\volume d\gggg\julynov24\Deep Learning\DA6401\TestHeatmaps\\' + filename + '.png')

    return attention_mat, pred_char_seq
