import os
import glob
import math
import time
import igraph as ig
import numpy as np
import numpy.matlib
import pandas as pd
import networkx as nx
from scipy.stats import entropy
from scipy.special import softmax
from collections import Counter
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations


def Renyi_entropy(pmf, order=1.3):
    pmf = pmf.flatten()
    if order == 0:
        H = np.log2(pmf.shape[0])
    elif order == 1:
        H = entropy(pmf, base=2)
    elif order == np.inf:
        H = -np.log2(pmf.max())
    else:
        H = 1/(1-order) * np.log2((pmf**order).sum())
    
    return H
    
    
def construct_full_JDAM_from_adj(adj, labels):
    if isinstance(labels, list):
        labels = np.array(labels)
    degree = adj.sum(axis=0)
    num_edges = adj.sum()
    degree_counter = Counter(degree)
    # remaining degree distribution of the network 
    qk = {deg - 1: degree_counter[deg] * deg / num_edges for deg in range(1, max(degree)+1)} # {degree - 1: degree * count / num_edges for degree, count in Counter(degree).items()}
    comat_full = np.zeros((len(qk)*2, len(qk)*2))

    rows, cols = adj.nonzero()
    for i in range(adj.shape[0]):
        label_i = labels[i]
        degree_i = degree[i]
        for j in cols[rows == i]: # for j in np.arange(adj.shape[1]):
            label_j = labels[j]
            degree_j = degree[j]
            comat_full[label_i * len(qk) + degree_i - 1, label_j * len(qk) + degree_j - 1] += adj[i, j]

    return comat_full
    
    
def MI_JDAM(comat_full, order=1.3):
    # Compute the mutual information of the joint degree distribution and the single degree distribution    
    n = comat_full.shape[0] // 2
    ejk = comat_full[:n] + comat_full[n:]
    ejk = ejk[:, :n] + ejk[:, n:]
    qk = ejk.sum(axis=0)
    I1 = Renyi_entropy(qk.reshape(-1, 1), order=order) * 2 - Renyi_entropy(ejk.flatten(), order=order) # entropy(ejk.flatten(), (qk.reshape(-1, 1) @ qk.reshape(1, -1)).flatten(), base=2)

    # Compute the mutual information of the joint (degree, attributes) distribution and the single degree distribution
    pjcj = comat_full.sum(axis=0)
    I2 = Renyi_entropy(pjcj.reshape(-1, 1), order=order) * 2 - Renyi_entropy(comat_full.flatten(), order=order) # entropy(comat_full.flatten(), (pjcj.reshape(-1, 1) @ pjcj.reshape(1, -1)).flatten(), base=2)

    return I2 - I1, I2, I1    
    
    
def attribute_assortativity(comat):
    length = comat.shape[0] // 2
    m = comat.sum()
    pcc = (comat[:length, :length] + comat[length:, length:]).sum() / m
    pc1 = comat[:length].sum() / m
    pc2 = comat[length:].sum() / m

    return (pcc - pc1**2 - pc2 ** 2) / (1 - pc1**2 - pc2 ** 2)


def degree_assortativity(comat):
    length = comat.shape[0] // 2
    ejk = comat[:length, :length] + comat[:length, length:] + comat[length:, length:] + comat[length:, :length]
    qk = ejk.sum(axis=0)
    m = ejk.sum()
    
    Eq = (np.arange(length) * qk / m).sum()
    Eq2 = (np.arange(length)**2 * qk / m).sum()
    Varq = Eq2 - Eq**2
    num = 0
    for j in range(length):
        for k in range(length):
            num += j * k * (ejk[j, k] / m - qk[j] * qk[k] / m**2)
    
    return num / Varq    
    

def metrics(g, node_labels, order=1, return_value=True, verbose=False):
    a1 = g.assortativity_degree()
    a2 = g.assortativity(node_labels)
    adj = np.array(g.get_adjacency().data)
    delta_I, I2, I1 = MI_JDAM(construct_full_JDAM_from_adj(adj, node_labels), order=order)
    if verbose:
        print(f"degree assortativity: {a1}, attribute assortativity: {a2}; \n I1: {I1}, I2: {I2}, dI: {delta_I}")
    if return_value:
        return delta_I, I2, I1, a1, a2
        
        
def plot_measure_correlation(aas, das, I1s, name="I1", order=1.3, savefig=False, coeff=True):
    markersize = 1
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    ax[0, 1].plot(aas, I1s, 'ko', markersize=markersize)
    ax[0, 0].plot(das, I1s, 'ko', markersize=markersize)
    ax[0, 1].set_title(r"$I(q; q')$ vs $\gamma_{att}$: " +r"$r={:.3f}$".format(np.corrcoef(aas, I1s)[0, 1]))
    ax[0, 0].set_title(r"$I(q; q')$ vs $\gamma_{deg}$: " +r"$r={:.3f}$".format(np.corrcoef(das[~np.isnan(das)], I1s[~np.isnan(das)])[0, 1]))
    ax[1, 1].plot(np.abs(aas), I1s, 'ko', markersize=markersize)
    ax[1, 0].plot(np.abs(das[~np.isnan(das)]), I1s[~np.isnan(das)], 'ko', markersize=markersize)
    ax[1, 1].set_title(r"$I(q; q')$ vs $|\gamma_{att}|$: " +r"$r={:.3f}$".format(np.corrcoef(np.abs(aas), I1s)[0, 1]))
    ax[1, 0].set_title(r"$I(q, q')$ vs $|\gamma_{deg}|$: " +r"$r={:.3f}$".format(np.corrcoef(np.abs(das[~np.isnan(das)]), I1s[~np.isnan(das)])[0, 1]))

    x = np.abs(das[~np.isnan(das)]); y = I1s[~np.isnan(das)]; x = x.reshape(-1, 1); y = y.reshape(-1, 1); Psi = np.hstack((x, np.ones_like(x)))
    beta = np.linalg.solve(Psi.T @ Psi, Psi.T @ y); x_hat = Psi @ beta
    ax[1, 0].plot(np.abs(das[~np.isnan(das)]), x_hat, 'r--', dashes=(3, 5))

    x = np.abs(aas); y = I1s; x = x.reshape(-1, 1); y = y.reshape(-1, 1); Psi = np.hstack((x, np.ones_like(x)))
    beta = np.linalg.solve(Psi.T @ Psi, Psi.T @ y); x_hat = Psi @ beta
    ax[1, 1].plot(np.abs(aas), x_hat, 'r--')

    plt.tight_layout()
    if savefig:
        fig.savefig(f"{name} correlation Renyi-{order}.png", dpi=100)
    if coeff:
        return np.corrcoef(np.abs(aas), I1s)[0, 1]
    

