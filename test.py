import networkx as nx
import matplotlib.pyplot as plt
from dnn import MDS
import torch
import torch.nn as nn
import torch.optim as optim

g = nx.Graph()

# Dominating set: {2, 5}
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)
g.add_edge(4, 5)

n = g.number_of_nodes()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = MDS(g).to(device)

# Define input
x = torch.ones(n, 1)

# Define minimum possible value for MDS dnn output
k = 2 # Best geuss at MDS size
goal = torch.zeros(1,1)
goal[0, 0] = -(n - k)/2

# Define the loss function
loss_fn = lambda predicted, desired: predicted - desired

# Initialize the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                last = param.clone()
# Training loop
for epoch in range(50000):
    optimizer.zero_grad()  # Clear previous gradients
    output = model(x)  # Forward pass
    loss = loss_fn(output, goal)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters

    if epoch % 5000 == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("******************* EPOCH " + str(epoch) + "*******************")
                print(param.data - last.data)
                print("Loss: " + str(loss))
                last = param.clone()


print("\n\n\n******************************\n\n RESULTS:\n")

################################
#  Display results on a graph  #
################################

thetas = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        thetas = param.data

S = []
S_not = []
for i in range(0, len(thetas)):
    if thetas[i] >= 0.99:
        S.append(i)
    else:
        S_not.append(i)

# Draw the graph
pos = nx.shell_layout(g)
nx.draw_networkx_nodes(g, pos, nodelist=S, node_color="tab:red")
nx.draw_networkx_nodes(g, pos, nodelist=S_not, node_color="tab:blue")
nx.draw_networkx_edges(g, pos)
plt.show()
