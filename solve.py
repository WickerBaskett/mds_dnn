# solve.py
# Elliott R. Lewandowski and Jonathan D. Brough
# 2025-12-03
# Generate dNN for MDS and print outputs
import networkx as nx
import matplotlib.pyplot as plt
from dnn import MDS
import torch
import torch.nn as nn
import torch.optim as optim

##############################
#  Input and Output Options  #
##############################

# Determines wether to print theta vector
print_theta = False

# Determines whether or not to display a visual graph
display_graph = True

# To solve a new graph change this!
g = nx.gnp_random_graph(10, 0.5)

#################################
#  Setup pytorch model for dNN  #
#################################

n = g.number_of_nodes()

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
model = MDS(g.copy()).to(device)

############################
#  Pass graph through dNN  #
############################

# Define input
x = torch.ones(n, 1)

# Pass through dNN
output = model(x)

####################
#  Display Results #
####################

print("\n******************************  RESULTS  ******************************\n")
print("dNN Output:\t" + str(output.data[0, 0].item()))

# Get theta values
thetas = []
for name, param in model.named_parameters():
    if print_theta:
        print(name, param.data)
    thetas = param.data

# Create groups of vertices to color the graph with
S = []
S_not = []
k = 0
for i in range(0, len(thetas)):
    if thetas[i] >= 0.99:
        k += 1
        S.append(i)
    else:
        S_not.append(i)

print("k:\t\t" + str(k))
print("n:\t\t" + str(n))
print("\n***********************************************************************\n")

# Draw the graph

if display_graph:
    pos = nx.shell_layout(g)
    nx.draw_networkx_nodes(g, pos, nodelist=S, node_color="tab:red")
    nx.draw_networkx_nodes(g, pos, nodelist=S_not, node_color="tab:blue")
    nx.draw_networkx_edges(g, pos)
    plt.show()
