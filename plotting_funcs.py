import matplotlib.pyplot as plt 
from datetime import datetime 

def plot_label_frequency(L):
    ABSTAIN = -1
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    figname = datetime.now().strftime('%Y%m%d%M') + 'label_freq.png'
    plt.savefig('outputs/' +figname)

def plot_probabilities_histogram(Y):
    plt.hist(Y, bins=10)
    plt.xlabel("Probability of POS")
    plt.ylabel("Number of data points")
    figname = datetime.now().strftime('%Y%m%d%M') + 'prob_hist.png'
    plt.savefig('outputs/' + figname)
