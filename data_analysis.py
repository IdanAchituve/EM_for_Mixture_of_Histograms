import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

topics = {"acq": 0, "money-fx": 1, "grain": 2, "crude": 3, "trade": 4, "interest": 5, "ship": 6, "wheat": 7, "corn": 8}

# calculate Perplexity and print graphs
def likelihood_and_perplexity(save_path, articles, log_likelihood):

    num_words = sum(utils.get_word_counts(articles).values())  # vocabulary size
    log_likelihood = np.asarray(log_likelihood)
    x_axis = np.asarray(list(range(len(log_likelihood))))

    # plot log likelihood
    path = save_path + "log_likelihood.png"
    plt.subplots_adjust(left=0.2)
    plt.plot(x_axis, log_likelihood)
    plt.title('Log-Likelihood (max value ' + str(round(log_likelihood[-1], 2)) + ")")
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iteration')
    plt.savefig(path)
    plt.close()

    # plot the perplexity
    path = save_path + "perplexity.png"
    perplexity = np.e**((-1/float(num_words)) * log_likelihood)
    print("Best Perplexity: " + str(perplexity[-1]))
    plt.title('Perplexity (min value ' + str(round(perplexity[-1], 2)) + ")")
    plt.plot(x_axis, perplexity)
    plt.ylabel('Perplexity')
    plt.xlabel('Iteration')
    plt.savefig(path)
    plt.close()


# generate confusion matrix
def confusion_matrix(path, w, articles, articles_headers):

    N = len(articles)  # number of documents
    clusters_per_doc = w.copy()

    # count documents associate with each cluster
    conf_matrix = np.zeros((9, 10))

    for doc in range(N):

        # for document t get the most probable clusters
        max_cluster = np.argmax(clusters_per_doc[doc, :])
        conf_matrix[max_cluster, -1] += 1

        # fill confusion matrix
        doc_topics = articles_headers[doc]  # get the topics of the document
        for doc_topic in doc_topics:
            doc_topic_idx = topics[doc_topic]
            conf_matrix[max_cluster, doc_topic_idx] += 1

    mat = pd.DataFrame(conf_matrix, columns=list(topics.keys()) + ["Number Of Articles"])
    sorted_mat = mat.sort_values("Number Of Articles", ascending=False)
    sorted_mat.to_csv(path)  # save confusion matrix

    return sorted_mat


# calculate accuracy
def calc_accuracy(w, articles, articles_headers, conf_matrix):

    conf_matrix.drop(columns=["Number Of Articles"], inplace=True)  # drop count column
    keys = list(conf_matrix.index)  # get clusters by names corresponding to w matrix
    values = list(conf_matrix.idxmax(axis=1))  # assign to cluster the most dominant topic
    cluster_to_class = dict(zip(keys, values))

    N = len(articles)  # number of documents
    clusters_per_doc = w.copy()

    accuracy = 0
    for doc in range(N):

        # for document t get the max cluster
        max_cluster = np.argmax(clusters_per_doc[doc, :])
        cluster_class = cluster_to_class[max_cluster]
        doc_topics = articles_headers[doc]  # get the topics of the document
        # increment if one of the document topics is the cluster topic
        if cluster_class in doc_topics:
            accuracy += 1

    print("Accuracy: " + str(float(accuracy)/N))


