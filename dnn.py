import torch
from torch import nn
import networkx as nx
from networkx import Graph

# Acknowledgments:
# Implementation of this was heavily inspired by the work of itsazibfarooq 
# who has published an implemntation of many dNN's on GitHub: 
#       https://github.com/itsazibfarooq/dataless

def heuristic(G):
    n = G.number_of_nodes()
    theta = torch.zeros(n)
    #deg = 
    while n != 0:
        pass

class Hadamard(nn.Module):
    """
    Trainable theta layer, does element wise multiplication with input
    Theta is restricted to range [0, 1]
    """
    def __init__(self, n):
        super().__init__()
        self.theta = nn.Parameter(torch.rand(n, 1) * (0.95 - 0.05) + 0.05, requires_grad=True)
        self.min_theta = 0
        self.max_theta = 1

    def forward(self, x):
        self.theta.data = self.theta.data.clamp(self.min_theta, self.max_theta)
        return x * self.theta

class Binary_Matrix(nn.Module):
    """
    Multiply by the binary matrix, this returns a n+m x 1 tensor
    """
    def __init__(self, W, b):
        super().__init__()
        self.W = W.T
        self.b = b
    
    def forward(self, x):
        return (self.W @ x) + self.b


class Weight_Vec(nn.Module):
    """
    Multiply by a fully connected weight vector to get final
    result. This will return a 1 x 1 tensor
    """
    def __init__(self, w):
        super().__init__()
        self.w = w.T
    
    def forward(self, x):
        return self.w @ x

### dNN for the MDDS problem, accepts a graph as an adjacency list
class MDS(nn.Module):
    """
    Implemntation of a dataless neural network to solve
    the Minimum Dominating Set problem
    """

    def __init__(self, graph: nx.graph):
        super().__init__()

        ###############################
        #  Generate Matrices/Vectors  #
        ###############################

        # Store infromation about the graph
        self.G = graph
        self.n = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()
        

        # Binary Matrix: First nxn is diagonal matrix
        #   second nxm has columns representing edges, 
        #   for each column there should be exactly 2 ones
        #   for the vertices associated with this edge and 
        #   everything else should be zero
        wshape = (self.n, 2 * self.n)
        self.W = torch.zeros(wshape)
        
        for i in range(0, self.n):
            self.W[i, i] = -1

        for node in self.G.nodes():
            self.W[node, self.n + node] = -1
            for neigh in  self.G.neighbors(node):
                self.W[neigh, self.n + node] = -1

        print("Binary Matrix: \n" + str(self.W))

        # Bias Vector: First n are 1/2 second n are 1
        self.b = torch.ones(2*  self.n, 1)

        for i in range(0, self.n):
            self.b[i] = 1/2

        print("Bias Vector:\n" + str(self.b))

        # Fully connected weight vector first n are -1, second n are -n
        self.w = torch.zeros(2 * self.n, 1)

        for i in range(0, self.n):
            self.w[i] = -1
        
        for j in range(self.n, 2 * self.n):
            self.w[j] = self.n

        print("Connected Weight Vector: \n" + str(self.w))

        ##################
        #  Setup Layers  #
        ##################

        # n x 1
        # e_0 hadamard theta
        self.layer1 = Hadamard(self.n)

        # Mult by W^T and apply bias vec, probably need to fix parameters here
        # ( 2n x n ) @ ( n x 1 ) = 2n x 1
        # W^T * layer1 + b
        self.layer2 = Binary_Matrix(self.W, self.b)

        # Apply ReLu
        # 2n x 1
        self.layer3 = nn.ReLU()

        # Mult with fully connected weight vector
        # (1 x 2n) @ (2n x 1) = 1 x 1
        self.layer4 = Weight_Vec(self.w)

    def forward(self, x):
        #print("\n\n******************\n")
        x = self.layer1(x)
        #print("Hadamard: " + str(x))
        x = self.layer2(x)
        #print("Binary Matrix: " + str(x))
        x = self.layer3(x)
        #print("ReLU: " + str(x))
        x = self.layer4(x)
        #print("Weight Vec: " + str(x))
        return x

