import os
import numpy as np
import utils
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import hnswlib
from scipy import spatial
import sys
import matplotlib.pyplot as plt
import lcc_cache
import ATTACKS
np.random.seed(0)


def knowledge_avg(knowledge, weights):
    result = []
    for k_ in knowledge:
        result.append(knowledge_avg_single(k_, weights))
    # return torch.Tensor(np.array(result)).cuda()
    return torch.Tensor(np.array(result))


def knowledge_avg_single(knowledge, weights):
    result = torch.zeros_like(knowledge[0]).cpu()
    sum = 0
    for _k, _w in zip(knowledge, weights):
        result.add_(_k.cpu() * _w)
        sum = sum + _w
    result = result / sum
    return torch.tensor(np.array(result.detach().cpu()))


class KnowledgeCache:
    def __init__(self, n_classes, R):
        self.n_classes = n_classes
        self.cache = {}
        self.idx_to_hash = {}
        self.relation = {}
        for i in range(n_classes):
            self.cache[i] = {}
        self.cache_R = R
        pass

    def output_relation(self):
        return self.relation
    def output_cache(self):
        return self.cache

    def add_hash(self, hash, label, idx):
        for k_, l_, i_ in zip(hash, label, idx):
            self.add_hash_single(k_, l_, i_)

    def add_hash_single(self, hash, label, idx):
        self.cache[int(label)][idx] = torch.Tensor(np.array([1.0 / self.n_classes for _ in range(self.n_classes)]))
        # {
        #     2: [torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])]
        # }
        self.idx_to_hash[idx] = hash

    def build_relation(self):
        hnsw_sim = 0
        for c in range(self.n_classes):
            idx_vectors = [key for key in self.cache[c].keys()]
            data = list()
            data = np.array([self.idx_to_hash[key].cpu().numpy() for key in idx_vectors])
            num_elements = data.shape[0]
            dim = data.shape[1]

            data_labels = np.arange(num_elements)

            index = hnswlib.Index(space='cosine', dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=1000, M=64)
            index.add_items(data, data_labels)
            index.set_ef(1000)
            labels, distances = index.knn_query(data, self.cache_R + 1)

            for idx, ele in enumerate(labels):
                self.relation[idx_vectors[int(idx)]] = []
                for x in ele[1:]:
                    self.relation[idx_vectors[int(idx)]].append(idx_vectors[x])

    def set_knowledge(self, knowledge, label, idx):
        for k_, l_, i_ in zip(knowledge, label, idx):
            self.set_knowledge_single(k_, l_, i_)

    def set_knowledge_single(self, knowledge, label, idx):
        self.cache[int(label)][idx] = knowledge

    def fetch_knowledge(self, label, idx):
        result = []
        for l_, i_ in zip(label, idx):
            result.append(self.fetch_knowledge_single(l_, i_))
        return result

    def fetch_knowledge_single(self, label, idx):
        result = []
        pairs = self.relation[idx]
        for pair in pairs:
            result.append(self.cache[int(label)][pair])
        return result

class PPVFedCache_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.global_logits_dict = dict()
        self.global_labels_dict = dict()
        self.global_extracted_feature_dict_test = dict()
        self.global_labels_dict_test = dict()
        self.criterion_KL = utils.KL_Loss(args.T)
        self.criterion_CE = F.cross_entropy

    def do_ppvfedcache_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                                train_data_local_dict, test_data_local_dict, args):
        image_scaler = transforms.Compose([
            transforms.Resize(224),
        ])
        print("*********start training with FedCache***************")

        train_data_local_dict_seq = {}
        train_data_num={}

        for client_index in range(args.client_number):
            train_data_local_dict_seq[client_index] = []
            for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                train_data_local_dict_seq[client_index].append((images, labels))
        knowledge_cache = KnowledgeCache(args.class_num, args.cache_R)
        encoder=mobilenet_v3_small(weights='IMAGENET1K_V1').cuda()
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
        encoder.eval()

        for client_index, client_model in enumerate(self.client_models):
            cur_idx = 0
            client_train_data_dist = [0 for _ in range(args.class_num)]
            for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                images, labels=images.cuda(), labels.cuda()
                batch_idx_list_label = list(labels)
                for i in batch_idx_list_label:
                    client_train_data_dist[i] += 1
                hash_code = encoder(image_scaler(images)).detach().cuda()
                hash_code = torch.tensor(hash_code.reshape((hash_code.shape[0], hash_code.shape[1])))
                for img, hash, label in zip(images, hash_code, labels):
                    knowledge_cache.add_hash_single(hash, label, (client_index, cur_idx))
                    cur_idx = cur_idx + 1
            train_data_num[client_index]=sum(client_train_data_dist)

        local_knowledge = {}
        local_knowledge_hash = {}
        global_knowledge = {}
        # 客户端本地知识的初始化
        for client_index in range(len(self.client_models)):
            local_knowledge_hash[client_index] = {}
            for c in range(args.class_num):
                local_knowledge_hash[client_index][c] = torch.Tensor(
                    np.array([1 / args.class_num for _ in range(args.class_num)]))
            local_knowledge_hash[client_index] = np.stack(
                [tensor.numpy() for tensor in local_knowledge_hash[client_index].values()], axis=0)
            for idx in range(train_data_num[client_index]):
                local_knowledge[(client_index, idx)] = torch.Tensor(
                    np.array([1 / args.class_num for _ in range(args.class_num)]))
                global_knowledge[(client_index, idx)] = torch.Tensor(
                    [1 / args.class_num for _ in range(args.class_num)])*args.R
        mal_idxs = utils.mal_select_clients(len(self.client_models), args.mal_rate)

        knowledge_cache.build_relation()

        for global_epoch in range(args.comm_round):
            print("*********communication round", global_epoch, "***************")
            metrics_all = {'test_loss': [], 'test_accTop1': [], 'test_accTop5': [], 'f1': []}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for client_index, client_model in enumerate(self.client_models):
                # print("*********start training on client", client_index, "***************")
                client_model=client_model.cuda()
                tmp_logits = {}
                for c in range(args.class_num):
                    tmp_logits[c] = []
                client_model.train()
                optim = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)
                cur_idx = 0
                for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                    images, labels = images.cuda(), labels.cuda().long()

                    log_probs = client_model(images)
                    if client_index in mal_idxs:
                        if 1 <= args.attack_method_id <= 5:
                            logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id)
                            log_probs = logits_attack.attack(log_probs)

                        if args.attack_method_id == 6:
                            tem_global_know=[global_knowledge[(client_index,idx)] for idx in range(train_data_num[client_index])]
                            tem = np.mean([tensor.detach().cpu().numpy() for tensor in tem_global_know],
                                          axis=0)
                            logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id,
                                                                    global_knowledge=tem)
                            log_probs = logits_attack.attack(log_probs)


                    loss_true = F.cross_entropy(log_probs, labels)

                    loss = None
                    teacher_knowledge = []
                    for img, logit, label in zip(images, log_probs, labels):
                        c = int(label)
                        tmp_logits[c].append(logit.detach().cpu().numpy())
                        teacher_knowledge.append(torch.Tensor((global_knowledge[(client_index,cur_idx)])/args.R))
                        local_knowledge[(client_index,cur_idx)]=logit
                        cur_idx =cur_idx + 1
                    teacher_knowledge = torch.stack(teacher_knowledge).cuda()

                    loss_kd = self.criterion_KL(log_probs, teacher_knowledge/ args.T)
                    loss = loss_true + args.alpha * loss_kd
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                for c in range(args.class_num):
                    if len(tmp_logits[c]) != 0:
                        local_knowledge_hash[client_index][c] = torch.mean(torch.Tensor(np.array(tmp_logits[c])),0)
                    else:
                        local_knowledge_hash[client_index][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)]))

            logical_graph = lcc_cache.LogicalGraphOracle(
                local_knowledges=local_knowledge_hash.copy(),
                mal_idxs=mal_idxs,
                args=args
            )
            client_connection, sim_matrix = logical_graph.logit_com_graph()

            sample_connect={}
            relation_origin = knowledge_cache.output_relation()

            for client_index_act,connections in client_connection.items():
                tem_sample=[]
                for idx in range(train_data_num[client_index_act]):
                    tem=[]
                    sample_con_orgin=relation_origin[(client_index_act,idx)]
                    for sample in sample_con_orgin:
                        if sample[0] in connections and len(tem)<args.R:
                            if sample[0] in mal_idxs and client_index_act not in mal_idxs:
                                tem_sample.append(sample[0])
                            tem.append(sample)
                    sample_connect[client_index_act, idx]=tem

            for idx in global_knowledge.keys():
                global_knowledge[idx] = torch.sum(torch.stack([local_knowledge[index].detach() for index in sample_connect[idx]],dim=0),dim=0).detach().cpu()

            if global_epoch % args.interval == 0:
                acc_top1_all = []
                acc_top5_all = []
                honest_top1_all = []
                honest_top5_all = []
                for client_index, client_model in enumerate(self.client_models):
                    client_model.cuda()
                    client_model.eval()
                    loss_avg = utils.RunningAverage()
                    accTop1_avg = utils.RunningAverage()
                    accTop5_avg = utils.RunningAverage()
                    for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                        images, labels = images.cuda(), labels.cuda().long()
                        log_probs = client_model(images)
                        loss = self.criterion_CE(log_probs, labels)
                        metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                        accTop1_avg.update(metrics[0].item())
                        accTop5_avg.update(metrics[1].item())
                        loss_avg.update(loss.item())
                    test_metrics = {str(client_index) + ' test_loss': loss_avg.value(),
                                    str(client_index) + ' test_accTop1': accTop1_avg.value(),
                                    str(client_index) + ' test_accTop5': accTop5_avg.value(),
                                    }

                    acc_top1 = accTop1_avg.value()
                    acc_top5 = accTop5_avg.value()
                    acc_top1_all.append(acc_top1)
                    acc_top5_all.append(acc_top5)
                    if client_index not in mal_idxs:
                        honest_top1_all.append(acc_top1)
                        honest_top5_all.append(acc_top5)

                print("mean Test/AccTop1 on all clients:", float(np.mean(np.array(honest_top1_all))))
