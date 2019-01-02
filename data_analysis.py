import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import utils

min_prob = 0.0000001

def likelihood_and_perplexity(articles, log_likelihood):

    os.makedirs("./graphs/", exist_ok=True)
    num_words = sum(utils.get_word_counts(articles).values())  # vocabulary size
    log_likelihood = np.asarray(log_likelihood)
    x_axis = np.asarray(list(range(len(log_likelihood))))

    # plot log likelihood
    path = "./graphs/log_likelihood.png"
    plt.tight_layout()
    plt.plot(x_axis, log_likelihood)
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iteration')
    plt.savefig(path)
    plt.close()

    # plot the perplexity
    path = "./graphs/perplexity.png"
    perplexity = np.e**((-1/float(num_words)) * log_likelihood)
    plt.plot(x_axis, perplexity)
    plt.ylabel('Perplexity')
    plt.xlabel('Iteration')
    plt.savefig(path)
    plt.close()


def confusion_matrix(path, w, articles, articles_headers):

    N = len(articles)  # number of documents
    clusters_per_doc = w.copy()
    # associate a document to a cluster if p(x_i|y_t) > min_prob
    clusters_per_doc[clusters_per_doc >= min_prob] = 1.0
    clusters_per_doc[clusters_per_doc < min_prob] = 0.0

    topics = {"acq":0, "money-fx":1, "grain":2, "crude":3, "trade":4, "interest":5, "ship":6, "wheat":7, "corn":8}

    # count documents associate with each cluster
    num_docs_in_clusters = np.sum(clusters_per_doc, axis=0)
    conf_matrix = np.zeros((9, 10))

    for doc in range(N):

        # for document t get all clusters
        rel_clusters = []
        for idx, cluster in enumerate(clusters_per_doc[doc, :]):
            if cluster == 1.0:
                rel_clusters.append(idx)

        # fill confusion matrix
        doc_topics = articles_headers[doc]  # get the topics of the document
        for doc_topic in doc_topics:
            doc_topic_idx = topics[doc_topic]
            for cluster in rel_clusters:
                conf_matrix[cluster, doc_topic_idx] += 1

    conf_matrix[:, -1] = num_docs_in_clusters  # on the last column put the number of documents
    mat = pd.DataFrame(conf_matrix, columns=list(topics.keys()) + ["Number Of Articles"])
    sorted_mat = mat.sort_values("Number Of Articles", ascending=False)
    sorted_mat.to_csv(path)