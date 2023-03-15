import numpy as np
import math

def calcu_cosine(vec1, vec2):
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

if __name__ == '__main__':
    dataset = 'pubmed'
    graph = open('graph/' + dataset + '_graph.txt', 'r', encoding='utf-8')
    feature = np.loadtxt('data/' + dataset + '.txt', dtype=float)
    similarity = open('similarity/' + dataset +'_cosine.txt', 'w', encoding='utf-8')
    for line in graph.readlines():
        n1 = line.strip().split()[0]
        n2 = line.strip().split()[1]
        vec1 = feature[int(n1)]
        vec2 = feature[int(n2)]
        cos_sim = calcu_cosine(vec1, vec2)
        if math.isnan(cos_sim):
            cos_sim = 0
        similarity.write(n1 + ' ' + n2 + ' ' + str(cos_sim) + '\n')


        



