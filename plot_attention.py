import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_attention(in_seq, out_seq, attentions):
    """ From http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"""

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([' '] + [str(x) for x in in_seq], rotation=90)
    ax.set_yticklabels([' '] + [str(x) for x in out_seq])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
