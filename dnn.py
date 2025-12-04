# dnn.py
# Elliott R. Lewandowski and Jonathan D. Brough
# 2025-12-03
# Pytorch implementation of MDS dNN with heuristic
# to generate initial theta
import torch
from torch import nn
import networkx as nx
from networkx import Graph

# Acknowledgments:
# Implementation of this was heavily inspired by the work of itsazibfarooq
# who has published a implemntations for a number of dNN's on GitHub:
#       https://github.com/itsazibfarooq/dataless


def heuristic(G: Graph):
    """
    Repeatdly find node with highest degree, add it
    to our solution set, and then remove it and its
    neighbors from the graph until no nodes are left

    :param G: Input Graph
    :type G: Graph
    :return: Theta tensor representing solution set
    :rtype: torch.Tensor
    """
    n = G.number_of_nodes()
    theta = torch.zeros(n, 1)

    while G.number_of_nodes() != 0:
        greatest = (-1, -1)
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


class Hadamard(nn.Module):
    """
    Trainable theta layer, does element wise multiplication with input
    Theta is restricted to range [0, 1]
    """

    def __init__(self, G: nx.Graph):
        super().__init__()
        self.theta = nn.Parameter(heuristic(G.copy()))
        self.min_theta = 0
        self.max_theta = 1

    def forward(self, x):
        self.theta.data = self.theta.data.clamp(self.min_theta, self.max_theta)
        return x * self.theta


class Binary_Matrix(nn.Module):
    """
    Multiply by the binary matrix, this returns a 2n x 1 tensor
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


class MDS(nn.Module):
    """
    Implemntation of a dataless neural network to solve
    the Minimum Dominating Set problem
    """

    def __init__(self, graph: nx.Graph):
        super().__init__()

        ###############################
        #  Generate Matrices/Vectors  #
        ###############################

        # Store infromation about the graph
        self.G = graph
        self.n = self.G.number_of_nodes()

        # Binary Matrix: First nxn is diagonal matrix
        #   second nxn has columns representing nodes,
        #   for each column there should be ones in
        #   every row corresponding to a node in the
        #   neighbourhood of our column node and in the
        #   row corresponding to the column node
        wshape = (self.n, 2 * self.n)
        self.W = torch.zeros(self.n, 2 * self.n)

        for i in range(0, self.n):
            self.W[i, i] = -1

        for node in self.G.nodes():
            self.W[node, self.n + node] = -1
            for neigh in self.G.neighbors(node):
                self.W[neigh, self.n + node] = -1

        # Bias Vector: First n are 1/2 second n are 1
        self.b = torch.ones(2 * self.n, 1)

        for i in range(0, self.n):
            self.b[i] = 1 / 2

        # Fully connected weight vector first n are -1, second n are -n
        self.w = torch.zeros(2 * self.n, 1)

        for i in range(0, self.n):
            self.w[i] = -1

        for j in range(self.n, 2 * self.n):
            self.w[j] = self.n

        ##################
        #  Setup Layers  #
        ##################

        # n x 1
        # e_0 hadamard theta
        self.layer1 = Hadamard(self.G)

        # Mult by W^T and apply bias vec
        # ( 2n x n ) @ ( n x 1 ) = 2n x 1
        # W^T @ layer1 + b
        self.layer2 = Binary_Matrix(self.W, self.b)

        # Apply ReLu
        # 2n x 1
        self.layer3 = nn.ReLU()

        # Mult with fully connected weight vector
        # (1 x 2n) @ (2n x 1) = 1 x 1
        self.layer4 = Weight_Vec(self.w)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
