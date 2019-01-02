import utils
import numpy as np
import data_analysis as da
import os

num_classes = 9
const = 1000000000
epsilon = 0.00001
k = 10
lidstone_lambda = 0.15


def log_likelihood(N, n, k, P, alpha):

    likelihood = 0
    # run over documents
    for d in range(N):
        z_i = []
        # compute z per class given document t
        for c in range(num_classes):
            z_ti = np.log(alpha[c]) + np.dot(n[d, :], np.log(P[c, :]))  # z_i = ln(alpha_i) + sigma_over_k(n_tk*ln(P_ik))
            z_i.append(z_ti)
        z_i = np.asarray(z_i)
        m_t = np.max(z_i)  # max z_ti

        # sum the exponents of elements that pass the threshold
        zi_minus_mi = 0
        for c in range(num_classes):
            if z_i[c] - m_t >= -k:
                zi_minus_mi += np.exp(z_i[c] - m_t)

        # lnL = sigma_over_t(m_t + sum(e^(z_ti-m_t)))
        likelihood += m_t + np.log(zi_minus_mi)
    return likelihood


# the E step
def E_step(N, n, k, P, alpha):

    # run over documents
    for d in range(N):
        z_i = []
        # compute z per class given document t
        for c in range(num_classes):
            z_ti = np.log(alpha[c]) + np.dot(n[d, :], np.log(P[c, :]))  # z_i = ln(alpha_i) + sigma_over_k(n_tk*ln(P_ik))
            z_i.append(z_ti)
        z_i = np.asarray(z_i)
        m = np.max(z_i)  # max z_ti

        # sum the exponents of elements that pass the threshold
        denominator = 0
        for c in range(num_classes):
            if z_i[c] - m >= -k:
                denominator += np.exp(z_i[c] - m)

        # take only elements that pass the threshold
        nominator = []
        for c in range(num_classes):
            if z_i[c] - m >= -k:
                nominator.append(np.exp(z_i[c] - m))
            else:
                nominator.append(0)

        w_ti = np.reshape(np.asarray(nominator)/denominator, (1, -1))
        w = w_ti if d == 0 else np.concatenate((w, w_ti))
    return w


# the M step
def M_step(N, V, n, w):

    # calc alpha_i per each class
    alpha = np.sum(w, axis=0)/N  # sum each column and divide by number of articles
    alpha[alpha == 0] = epsilon  # in case a class has probability of 0
    alpha /= np.sum(alpha)  # correct all values to get proper probability distribution

    # calc P(w_k|C_i)
    n_t = np.sum(n, axis=1)  # sum frequencies per document. shape: Nx1
    for idx, weight_ti in enumerate(w.T):
        help_mat = np.reshape(weight_ti, (-1, 1)) * n  # per each word and document for class i calc: w_ti*n_tk.  Shape: NxV
        nominator = np.sum(help_mat, axis=0) + lidstone_lambda  # sum each column: sigma_over_t(w_ti*n_tk) + lambda. Shape: Vx1
        denominator = np.dot(weight_ti, n_t) + V*lidstone_lambda  # sigma_over_t(w_ti*n_t) + V*lambda
        P_i = np.reshape(nominator/denominator, (1, -1))  # get P_ik for all words. Shape: 1xV
        P = P_i if idx == 0 else np.concatenate((P, P_i), axis=0)  # shape: CxV

    return P, alpha


# EM schema
def EM(articles):

    N = len(articles)  # number of documents
    V = len(utils.get_word_counts(articles).keys())  # vocabulary size
    n = utils.words_per_doc_vec(articles)  # shape of NxV, each cell in n_tk
    likelihod_vals = []

    # initial weights - each example belong to 1 cluster only
    w = np.zeros((N, num_classes))
    for idx in range(N):
        w[idx, idx % num_classes] = 1

    P, alpha = M_step(N, V, n, w)
    likelihood = log_likelihood(N, n, k, P, alpha)
    likelihod_vals.append(likelihood)

    # do the EM until convergence
    eps = abs(likelihood/const)
    delta = abs(likelihood)
    iter = 0
    while delta > eps:
        w = E_step(N, n, k, P, alpha)
        P, alpha = M_step(N, V, n, w)
        likelihood = log_likelihood(N, n, k, P, alpha)
        likelihod_vals.append(likelihood)
        delta = likelihod_vals[-1] - likelihod_vals[-2]
        print("log-likelihood " + "iteration " + str(iter) + ": " + str(likelihood))
        iter += 1

    return likelihod_vals, w


if __name__ == '__main__':

    dev_set = "./dataset/develop.txt"
    save_path = "./output_idan_300083029/"
    os.makedirs(save_path, exist_ok=True)

    articles, articles_headers = utils.read_data(dev_set)  # read data
    articles_filtered = utils.remove_rare_words(articles)

    likelihood, w = EM(articles_filtered)
    da.likelihood_and_perplexity(save_path, articles_filtered, likelihood)
    conf_matrix = da.confusion_matrix(save_path + "confusion_matrix.csv", w, articles_filtered, articles_headers)
    da.calc_accuracy(w, articles_filtered, articles_headers, conf_matrix)