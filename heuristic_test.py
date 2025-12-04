import networkx as nx
import matplotlib.pyplot as plt
from dnn import MDS
import torch
import torch.nn as nn
import torch.optim as optim

def heuristic(G: nx.Graph):
    n = G.number_of_nodes()
    theta = torch.zeros(n)

    while G.number_of_nodes() != 0:
        greatest = (-1,-1)
        for node in G.degree():
            if node[1] > greatest[1]:
                greatest = node

        theta[greatest[0]] = 1
        neighbors = []
        for neigh in G.neighbors(greatest[0]):
            neighbors.append(neigh)
        for n in neighbors:
            G.remove_node(n)
        G.remove_node(greatest[0])



    return theta

g = nx.path_graph(25)

theta = heuristic(g.copy())
print(theta)

S = []
S_not = []
for i in range(0, len(theta)):
    if theta.data[i] >= 0.99:
        S.append(i)
    else:
        S_not.append(i)

# Draw the graph
pos = nx.planar_layout(g)
nx.draw_networkx_nodes(g, pos, nodelist=S, node_color="tab:red")
nx.draw_networkx_nodes(g, pos, nodelist=S_not, node_color="tab:blue")
nx.draw_networkx_edges(g, pos)
plt.show()
