import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from k_means import K_Means

def get_cluster_entropy(n_healthy, n_diseased):
    if n_healthy == 0:
        counts = np.asarray([n_diseased])
    elif n_diseased == 0:
        counts = np.asarray([n_healthy])
    else:
        counts = np.asarray([n_healthy, n_diseased])

    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def get_results(k):
    k_means = K_Means(k=k)
    cluster_labels = k_means.predict(myData)
    clusters = k_means.clusters
    general_percentage_clusters = [[] for _ in range(k)]

    for cluster_label in range(k):
        n_healthy = 0
        n_diseased = 0
        for data_idx in clusters[cluster_label]:
            label = labels[data_idx]
            if label == 1:
                n_healthy += 1
            elif label == 0:
                n_diseased += 1

        healthy_percentage = n_healthy/len(clusters[cluster_label])  # for cluster label #
        disease_percentage = n_diseased/len(clusters[cluster_label])  # for cluster label #

        general_percentage = (n_diseased + n_healthy) /(disease_sample_count + healthy_sample_count)
        general_percentage_clusters[cluster_label].append(general_percentage)
        general_percentage_clusters[cluster_label].append([healthy_percentage, disease_percentage])
        general_percentage_clusters[cluster_label].append(n_healthy)
        general_percentage_clusters[cluster_label].append(n_diseased)
        general_percentage_clusters[cluster_label].append(get_cluster_entropy(n_healthy,n_diseased))

    return general_percentage_clusters


def get_optimized_results(k=2, n_trials=50):

    comp_results = [[], [], []]  # max percentage, seed, clusters

    for trial_number in range(n_trials):
        seed = np.random.randint(n_trials * 100)
        np.random.seed(seed)
        general_percentage_clusters = get_results(k)
        healthy_comp = []
        for n_cluster in general_percentage_clusters:
            healthy_comp.append(n_cluster[1][0])
        max_healthy_percentage = max(healthy_comp)
        comp_results[0].append(max_healthy_percentage)
        comp_results[1].append(seed)
        comp_results[2].append(general_percentage_clusters)

        print("")
        print("Trial: ", trial_number)
        print("seed: ", seed)
        print("max_healthy_percentage: ", max_healthy_percentage)

    _max_percent = max(comp_results[0])
    best_index = comp_results[0].index(max(comp_results[0]))

    print("Overall max percentage:", _max_percent, " seed:", comp_results[1][best_index])
    print("Plotting pie chart...")

    general_sizes_pie = []
    general_sizes_str=[]
    entropy_values = []
    for _clusters in comp_results[2][best_index]:
        _general_percentage = round(_clusters[0]*100, 5)
        general_sizes_pie.append(_general_percentage)
        h_perc = round(_clusters[1][0]*100, 3)
        d_perc = round(_clusters[1][1]*100, 3)
        h_count = _clusters[2]
        d_count = _clusters[3]
        p_entropy = (h_count + d_count)/disease_sample_count
        v_entropy = (p_entropy * _clusters[4])
        entropy_values.append(v_entropy)
        my_str = ' Healthy:' + str(h_perc) + '%' + "({}),".format(h_count) +\
                 ' Diseased:' + str(d_perc) + "%" + "({})".format(d_count)
        general_sizes_str.append(my_str)

    general_entropy = sum(entropy_values)

    plt.title('K = ' + str(k)+', General Entropy: ' + str(round(general_entropy, 5)))
    patches, texts, juck = plt.pie(general_sizes_pie, startangle=90, radius=1, shadow=True, autopct='%10.1f%%')
    plt.legend(patches, general_sizes_str, loc="lower left", fontsize='large')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return comp_results


myData = pd.read_excel('dataset/DifferentiallyExpressedGenes.xlsx', index_col=None, header=None)
print(myData)
myData = myData.drop(myData.columns[0], axis=1).T
print(myData)
labels = myData[myData.columns[0]].values

print(labels)

healthy_sample_count=0
for _label in labels:
    if _label ==1:
        healthy_sample_count += 1


print('#####healthy_sample_count######')
print(healthy_sample_count)
disease_sample_count = len(labels)-healthy_sample_count
print('#####disease_sample_count######')
print(disease_sample_count)
myData = myData.drop(myData.columns[0], axis=1).values

results = get_optimized_results(k=7, n_trials=10)