import numpy as np
import torch
import torch.nn as nn
import random
from torch.nn import functional as F
import torch
class LogitsProcessor:
    def __init__(self,attack_method_id,global_knowledge):
        self.attack_method_id=attack_method_id
        self.global_knowledge=global_knowledge

    def attack(self, log_probs):
        if self.attack_method_id == 1:
            # print("FDLA")
            log_probs= self.FDLA_logits(log_probs)
        elif self.attack_method_id == 2:
            # print("PCFDLA")
            log_probs=self.PCFDLA_logits(log_probs)
        elif self.attack_method_id == 3:
            # print("FDPLA")
            log_probs= self.FDPLA_logits(log_probs)
        elif self.attack_method_id == 4:
            # print("FDFXA")
            log_probs= self.Fixed_logits(log_probs)
        elif self.attack_method_id == 5:
            # print("FDRMA")
            log_probs= self.Random_logits(log_probs)
        elif self.attack_method_id == 6:
            log_probs=self.FDSA_logits(log_probs)
        return log_probs

    def FDSA_logits(self, log_probs):
        regression_factor = 0.5
        scale=2
        mean_vector_glo = np.linalg.norm(np.mean(self.global_knowledge, axis=0))
        mean_vector_loc = torch.norm(log_probs.mean(dim=0),p=2)
        scale_factor = max(scale*mean_vector_glo/mean_vector_loc,scale)
        attacked_logits = log_probs.clone()
        batch_size, num_classes = attacked_logits.shape

        for i in range(batch_size):

            row = attacked_logits[i]
            max_val, max_idx = torch.max(row, dim=0)
            random_idx = random.choice([idx for idx in range(len(row)) if idx != max_idx])
            random_val = row[random_idx]
            row[max_idx] = random_val
            row[random_idx] = max_val
            mean_val = (row[max_idx] + row[random_idx]) / 2
            row[max_idx] = row[max_idx] + regression_factor * (mean_val - row[max_idx])
            row[random_idx] = row[random_idx] + regression_factor * (mean_val - row[random_idx])
            attacked_logits[i] = row
        attacked_logits = attacked_logits * scale_factor
        return attacked_logits


    def FDLA_logits(self, log_probs):
        num_rows, num_columns = log_probs.shape
        changed_logits = torch.empty_like(log_probs)
        for i in range(num_rows):
            sorted_indices = torch.argsort(log_probs[i], descending=True)
            sorted_values = log_probs[i][sorted_indices]
            changed_indices = torch.cat((sorted_indices[1:], sorted_indices[:1]))
            changed_logits[i].scatter_(0, changed_indices, sorted_values)
        return changed_logits

    def FDPLA_logits(self, log_probs):
        num_rows, num_columns = log_probs.shape
        changed_logits = torch.empty_like(log_probs)
        for i in range(num_rows):
            ascending_indices = torch.argsort(log_probs[i], descending=False)
            descending_indices = torch.argsort(log_probs[i], descending=True)

            sorted_ascending = log_probs[i][ascending_indices]
            sorted_descending = log_probs[i][descending_indices]
            changed_logits[i].scatter_(0, descending_indices, sorted_ascending)

        return changed_logits

    def PCFDLA_logits(self, log_probs):
        s = torch.FloatTensor(1).uniform_(-20, 20).cuda().requires_grad_()
        indices = torch.argsort(log_probs, dim=1)
        output = -s * torch.ones_like(log_probs)
        for i in range(log_probs.size(0)):
            second_highest_index = indices[i, -2]
            output[i, second_highest_index] = s
        return output

    def Fixed_logits(self, log_probs):
        num_columns = log_probs.size(1)
        random_values = torch.FloatTensor(num_columns).uniform_(-20, 20)
        for i in range(num_columns):
            log_probs[:, i] = random_values[i]
        return log_probs

    def Random_logits(self, log_probs):
        mean = 1.0
        std = 10.0
        random_logits = torch.normal(mean=mean, std=std, size=log_probs.size())
        log_probs.data.copy_(random_logits)
        return log_probs


