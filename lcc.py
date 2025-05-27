import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import torch

class LogicalGraphOracle():
    def __init__(
        self,
        local_knowledges,
        mal_idxs,
        args,
    ):
        self.local_knowledges=local_knowledges
        self.args=args
        self.mal_idsx=mal_idxs
        self.hashed_logits()
        self.sim_matrix()

    def hashed_logits(self):
        print("*********start distributing hash_global_logits with FD***************")
        self.global_hashed_logits = {}
        for client_index in range(self.args.client_number):
            self.global_hashed_logits[client_index] = self.local_knowledges[client_index]
    def sim_matrix(self):
        self.similarity_matrix = np.zeros((self.args.client_number, self.args.client_number))
        for i in range(self.args.client_number):
            for j in range(i, self.args.client_number):
                if i != j:
                    A = self.global_hashed_logits[i].flatten()
                    B = self.global_hashed_logits[j].flatten()
                    cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8)
                    self.similarity_matrix[i, j] = cosine_sim
                    self.similarity_matrix[j, i] = cosine_sim
            self.similarity_matrix[i,i]=-1

    def logit_com_graph(self):
        intimacy_table = {i: np.argsort(-self.similarity_matrix[i]) for i in range(self.args.client_number)}
        neighbors_act = {i: intimacy_table[i][:self.args.R] for i in range(self.args.client_number)}
        return neighbors_act,self.similarity_matrix