import utils
import numpy as np

MAX = 999999999
epsilon = 0.00001
num_classes = 9


# the E step
def E_step(N, V, n, P, alpha):

    for d in range(N):
        for c in range(num_classes):
            z_ti = np.log(alpha[c]) + np.dot(n[d,:], np.log(P[c,:]))  # z_i = ln(alpha_i) + sigma_over_k(n_tk*ln(P_ik))
            z =


# the M step
def M_step(N, V, n, weights, lidstone_lambda):

    # calc alpha_i per each class
    alpha = np.sum(weights, axis=0)/N  # sum each column and divide by number of articles
    alpha[alpha == 0] = epsilon  # in case a class has probability of 0
    alpha /= np.sum(alpha)  # correct all values to get proper probability distribution

    # calc P(w_k|C_i)
    n_t = np.sum(n, axis=1)  # sum frequencies per document. shape: Nx1
    for idx, weight_ti in enumerate(weights.T):
        help_mat = np.reshape(weight_ti, (-1, 1)) * n  # per each word and document for class i calc: w_ti*n_tk.  Shape: NxV
        nominator = np.sum(help_mat, axis=0) + lidstone_lambda  # sum each column: sigma_over_t(w_ti*n_tk) + lambda. Shape: Vx1
        denominator = np.dot(weight_ti, n_t) + V*lidstone_lambda  # sigma_over_t(w_ti*n_t) + V*lambda
        P_i = np.reshape(nominator/denominator, (1, -1))  # get P_ik for all words. Shape: 1xV
        P = P_i if idx == 0 else np.concatenate((P, P_i), axis=0)  # shape: CxV

    return P, alpha


# EM schema
def EM(articles, lidstone_lambda=0.02, k=10):

    N = len(articles)  # number of documents
    V = len(utils.get_word_counts(articles).keys())  # vocabulary size
    n = utils.words_per_doc_vec(articles)  # shape of NxV, each cell in n_tk

    # initial weights - each example belong to 1 cluster only
    weights = np.zeros((N, num_classes))
    for idx in range(N):
        weights[idx, idx % num_classes] = 1

    M_step(N, V, n, weights, lidstone_lambda)
    #while delta > epsilon:



if __name__ == '__main__':

    dev_set = "./dataset/develop.txt"
    articles, articles_headers = utils.read_data(dev_set)  # read data
    articles_filtered = utils.remove_rare_words(articles)

    EM(articles_filtered, 0.02, 10)