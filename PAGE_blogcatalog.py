# encoding=utf8
import numpy as np
import networkx as nx

import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

import argparse
import time


class PAGE():
    def __init__(self, graph_file, dimension):
        self.graph = graph_file

        self.dimension = dimension

        self.G = nx.read_edgelist(self.graph, nodetype=int, create_using=nx.DiGraph())
        self.G = self.G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))

        for e in self.G.edges():
            if e[0] != e[1]:
                matrix0[e[0], e[1]] = 1
                matrix0[e[1], e[0]] = 1
        self.matrix0 = scipy.sparse.csr_matrix(matrix0)
        print(matrix0.shape)

    def get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
        Sigma, U = sp.linalg.eigs(smat, k=self.dimension)

        print('sparsesvd time', time.time() - t1)
        return U, Sigma

    def compute_base(self, A):
        # NE Enhancement via Spectral Propagation
        print('Compute Base -----------------')
        t1 = time.time()

        DADA = self.normalize_adj(A)
        Base = 0.5*(sp.eye(self.node_number) + DADA)
          
        return Base

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""

        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsc()    


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProNE.")
    parser.add_argument('-graph', nargs='?', default='data/blogcatalog.ungraph',
                        help='Graph path')

    parser.add_argument('-dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('-label', nargs='?', default='data/blogcatalog.cmty',
                        help='Input label file path')

    parser.add_argument('-shuffle', type=int, default=10,
                        help='number of shuffule')
    return parser.parse_args()


#%%

import argparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from collections import defaultdict
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")
import sys

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sparse.lil_matrix(probs.shape)
        
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i,label] = 1
        return all_labels


def load_labels(labels_file, nodesize):
    # load label from label file, which each line i contains all node who have label i
    with open(labels_file) as f:
        context = f.readlines()
        print('class number: ', len(context))
        label = sparse.lil_matrix((nodesize, len(context)))

        for i, line in enumerate(context):
            line = map(int,line.strip().split('\t'))
            for node in line:
                label[node, i] = 1
    return label


def evaluate(features_matrix):
    args = parse_args()

    print(features_matrix.shape)
    nodesize = features_matrix.shape[0]
    label_matrix = load_labels(args.label, nodesize)
    number_shuffles = args.shuffle
    
    shuffles = []
    for x in range(number_shuffles):
          shuffles.append(skshuffle(features_matrix, label_matrix))

    all_results = defaultdict(list)

    training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for train_percent in training_percents:
        
        print(train_percent)
        for shuf in shuffles:
            
            X, y = shuf
            training_size = int(train_percent * nodesize)

            X_train = X[:training_size, :]
            y_train = y[:training_size, :]

            X_test = X[training_size:, :]
            y_test = y[training_size:,:]

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro", "samples", "weighted"]
            for average in averages:
                results[average] = f1_score(y_test,  preds, average=average)

            all_results[train_percent].append(results)
    print('Results, using embeddings of dimensionality', X.shape[1])
    print('-------------------')
    print('Train percent:', 'average f1-score')
    for train_percent in sorted(all_results.keys()):
        av = 0
        stder = np.ones(number_shuffles)
        i = 0
        for x in all_results[train_percent]:
            stder[i] = x["micro"]
            i += 1
            av += x["micro"]
        av /= number_shuffles
        print(train_percent, ":", av)



args = parse_args()

t_0 = time.time()
model = PAGE(args.graph, args.dimension)
t_1 = time.time()
print('Construct model', t_1 - t_0)

M = model.compute_base(model.matrix0)
t_2 = time.time()

print('Compute Base', t_2 - t_1)

vecs, vals = model.get_embedding_rand(M)
t_3 = time.time()


vecs = vecs.real  
vals = vals.real

print('Spectral Pro time', t_3 - t_2)

vals = 1.0*vals

d =128

 
M = M.todense()
NumberOfNodes = model.node_number

synapses0 = np.zeros((d,NumberOfNodes))

synapses0[:,:] = vecs.T

eigenvalues = vals.real
   
Nc=7
N=NumberOfNodes
Ns=NumberOfNodes

hid=128    # number of hidden units that are displayed in Ky by Kx array
mu=0.0
sigma=0.1
Nep=1     # number of epochs
Num=NumberOfNodes      # size of the minibatch
prec=1e-30
delta=0   # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=1          # ranking parameter, must be integer that is bigger or equal than 2



inputs=M
 
tot_input=np.dot(synapses0,inputs)

y=np.argsort(np.abs(tot_input),axis=0)
yl=np.zeros((hid,Num))
yl[y[0:k,:],np.arange(Num)]=1.0  # From low to high, the number indexes higher value

print(np.sum(yl))

Activation0 = np.true_divide(np.sum(yl,1), NumberOfNodes/hid)+0.5

Activation0[Activation0>1.1] = 1.1

t_4 = time.time()

print('Derive activation time', t_4 - t_3)

factor = np.sign(eigenvalues)*np.power(np.abs(eigenvalues), Activation0)
factor = np.diag(factor)

synapses0 = np.dot(factor, synapses0)

Rep1 = synapses0.T

Rep = preprocessing.normalize(Rep1, 'l2')


evaluate(Rep)