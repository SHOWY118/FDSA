import os
import utils
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import ATTACKS
import lcc

def tensor_cross_entropy(output: torch.Tensor, target: torch.Tensor):
    return -1.0 * (output.log() * target).mean()

def knowledge_avg(knowledge, weights):
    result = []
    for k_ in knowledge:
        result.append(knowledge_avg_single(k_, weights))
    return torch.Tensor(np.array(result)).cuda()

def knowledge_avg_single(knowledge, weights):
    result = torch.zeros_like(knowledge[0].knowledge)
    sum = 0
    for _k, _w in zip(knowledge, weights):
        result = result + _k.knowledge * _w
        sum = sum + _w
    result = result / sum
    return np.array(result.detach().cpu())

class PPVFD_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.criterion_KL = utils.KL_Loss(args.T)
        self.criterion_CE = F.cross_entropy

    def do_ppvfd_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                          train_data_local_dict, test_data_local_dict, args):

        print("*********start initializing with FD***************")

        global_knowledge = {}
        local_knowledge = {}

        for client_index in range(len(self.client_models)):
            global_knowledge[client_index] = {}
            local_knowledge[client_index] = {}
            for c in range(args.class_num):
                global_knowledge[client_index][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)])) * args.client_number
                local_knowledge[client_index][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)]))
            local_knowledge[client_index] = np.stack(
                [tensor.numpy() for tensor in local_knowledge[client_index].values()],
                axis=0)
            global_knowledge[client_index] = np.stack(
                [tensor.numpy() for tensor in global_knowledge[client_index].values()],
                axis=0)

        mal_idxs=utils.mal_select_clients(len(self.client_models),args.mal_rate)

        print("*********start training with FD***************")
        for global_epoch in range(args.comm_round):
            metrics_all = {'test_loss': [], 'test_accTop1': [], 'test_accTop5': [], 'f1': []}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for client_index, client_model in enumerate(self.client_models):
                tmp_logits = {}
                for c in range(args.class_num):
                    tmp_logits[c] = []
                client_model=client_model.cuda()
                client_model.train()
                optim = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)

                for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                    images, labels = images.cuda(), labels.cuda().long()
                    log_probs = client_model(images)
                    if client_index in mal_idxs:
                        if 1 <= args.attack_method_id <= 5:
                            logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id,global_knowledge=global_knowledge[client_index])
                            log_probs = logits_attack.attack(log_probs)
                        if args.attack_method_id==6:
                            logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id,global_knowledge=global_knowledge[client_index]/(args.R+1))
                            log_probs = logits_attack.attack(log_probs)

                    loss_true = F.cross_entropy(log_probs, labels)
                    soft_label = []
                    for logit, label in zip(log_probs, labels):
                        c = int(label)
                        soft_label.append((global_knowledge[client_index][c] - local_knowledge[client_index][c]) /args.R )
                        tmp_logits[c].append(logit.cpu().detach().numpy())
                    soft_label= torch.stack(
                        [torch.from_numpy(item) if isinstance(item, np.ndarray) else item for item in soft_label]
                    ).float().cuda()
                    # loss_kd = F.cross_entropy(log_probs, F.softmax(soft_label))
                    loss_kd= self.criterion_KL(log_probs, soft_label/args.T)
                    loss = loss_true + args.alpha * loss_kd
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                for c in range(args.class_num):
                    if len(tmp_logits[c]) != 0:
                        local_knowledge[client_index][c] = torch.mean(torch.Tensor(np.array(tmp_logits[c])),0)
                    else:
                        local_knowledge[client_index][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)]))

            logical_graph=lcc.LogicalGraphOracle(
                local_knowledges=local_knowledge.copy(),
                mal_idxs=mal_idxs,
                args=args
            )
            client_connection,sim_matrix=logical_graph.logit_com_graph()

            #更新全局知识
            client_connections = {node: sorted(list(client_connection[node]) + [node]) for node in
                                       [i for i in range(args.client_number)]}
            print(len(client_connections[0]))
            for client_index in range(args.client_number):
                global_knowledge[client_index]= sum([local_knowledge[index] for index in client_connections[client_index]])

            # print("*********start verifying with FedD***************")
            if global_epoch % args.interval == 0:
                acc_top1_all = []
                acc_top5_all = []
                honest_top1_all=[]
                honest_top5_all=[]
                for client_index, client_model in enumerate(self.client_models):
                    if client_index % args.sel != 0:
                        continue
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

                    # compute mean of all metrics in summary
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


                print("mean Test/AccTop1 on all honest clients:", float(np.mean(np.array(honest_top1_all))))
                print("mean Test/AccTop5 on all honest clients:", float(np.mean(np.array(honest_top5_all))))
