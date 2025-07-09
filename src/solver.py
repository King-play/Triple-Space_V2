import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)


    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        if self.train_config.data == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.MSELoss(reduction="mean")


        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.semi_domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")  # 新增
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        
        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        
        train_losses = []
        valid_losses = []
        for e in range(self.train_config.n_epoch):
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss_semi = []  # 新增
            train_loss = []
            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                batch_size = t.size(0)
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                
                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                semi_domain_loss = self.get_semi_domain_loss()  # 新增
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss
                
                loss = cls_loss + \
                    self.train_config.diff_weight * diff_loss + \
                    self.train_config.sim_weight * similarity_loss + \
                    self.train_config.recon_weight * recon_loss + \
                    self.train_config.sim_weight * semi_domain_loss  # 新增半公共域损失

                loss.backward()
                
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss_semi.append(semi_domain_loss.item())  # 新增
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())
                

            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc = self.eval(mode="dev")
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        self.eval(mode="test", to_print=True)



    
    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))
            

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                
                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """


        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            
            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            
            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
            
            return accuracy_score(binary_truth, binary_preds)


    def get_domain_loss(self,):

        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_semi_domain_loss(self,):
        """
        半公共空间域对抗损失：
        - TV半公共：T和V应该无法被区分，A应该能被区分出来
        - TA半公共：T和A应该无法被区分，V应该能被区分出来  
        - VA半公共：V和A应该无法被区分，T应该能被区分出来
        """
        loss = 0.0

        # TV半公共空间：期望T=0, V=0, A=2（A应该被识别为不同类别）
        semi_pred_tv = torch.cat((
            self.model.semi_domain_label_tv_t,  # T
            self.model.semi_domain_label_tv_v,  # V  
            self.model.semi_domain_label_tv_a   # A (来自其他空间)
        ), dim=0)
        
        batch_size = self.model.semi_domain_label_tv_t.size(0)
        semi_true_tv = to_gpu(torch.LongTensor([0]*batch_size + [0]*batch_size + [2]*batch_size))
        
        loss += self.semi_domain_loss_criterion(semi_pred_tv, semi_true_tv)

        # TA半公共空间：期望T=0, A=1, V=2
        semi_pred_ta = torch.cat((
            self.model.semi_domain_label_ta_t,  # T
            self.model.semi_domain_label_ta_a,  # A
            self.model.semi_domain_label_ta_v   # V (来自其他空间)
        ), dim=0)
        
        semi_true_ta = to_gpu(torch.LongTensor([0]*batch_size + [1]*batch_size + [2]*batch_size))
        
        loss += self.semi_domain_loss_criterion(semi_pred_ta, semi_true_ta)

        # VA半公共空间：期望V=1, A=1, T=2  
        semi_pred_va = torch.cat((
            self.model.semi_domain_label_va_v,  # V
            self.model.semi_domain_label_va_a,  # A
            self.model.semi_domain_label_va_t   # T (来自其他空间)
        ), dim=0)
        
        semi_true_va = to_gpu(torch.LongTensor([1]*batch_size + [1]*batch_size + [2]*batch_size))
        
        loss += self.semi_domain_loss_criterion(semi_pred_va, semi_true_va)

        return loss / 3.0

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # 原有的全共享空间相似性损失
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        
        # 半公共空间内部相似性损失
        loss += self.loss_cmd(self.model.utt_semi_shared_tv_t, self.model.utt_semi_shared_tv_v, 5)
        loss += self.loss_cmd(self.model.utt_semi_shared_ta_t, self.model.utt_semi_shared_ta_a, 5)
        loss += self.loss_cmd(self.model.utt_semi_shared_va_v, self.model.utt_semi_shared_va_a, 5)
        
        return loss / 6.0

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # 半公共表示
        semi_shared_tv_t = self.model.utt_semi_shared_tv_t
        semi_shared_tv_v = self.model.utt_semi_shared_tv_v
        semi_shared_ta_t = self.model.utt_semi_shared_ta_t
        semi_shared_ta_a = self.model.utt_semi_shared_ta_a
        semi_shared_va_v = self.model.utt_semi_shared_va_v
        semi_shared_va_a = self.model.utt_semi_shared_va_a

        loss = 0.0

        # 原有的私有-共享差异损失
        loss += self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # 原有的跨私有差异损失
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        # 私有与半公共的差异损失
        loss += self.loss_diff(private_t, semi_shared_tv_t)
        loss += self.loss_diff(private_t, semi_shared_ta_t)
        loss += self.loss_diff(private_v, semi_shared_tv_v)
        loss += self.loss_diff(private_v, semi_shared_va_v)
        loss += self.loss_diff(private_a, semi_shared_ta_a)
        loss += self.loss_diff(private_a, semi_shared_va_a)

        # 全共享与半公共的差异损失
        loss += self.loss_diff(shared_t, semi_shared_tv_t)
        loss += self.loss_diff(shared_t, semi_shared_ta_t)
        loss += self.loss_diff(shared_v, semi_shared_tv_v)
        loss += self.loss_diff(shared_v, semi_shared_va_v)
        loss += self.loss_diff(shared_a, semi_shared_ta_a)
        loss += self.loss_diff(shared_a, semi_shared_va_a)

        # 不同半公共空间之间的差异损失
        loss += self.loss_diff(semi_shared_tv_t, semi_shared_ta_t)
        loss += self.loss_diff(semi_shared_tv_v, semi_shared_va_v)
        loss += self.loss_diff(semi_shared_ta_a, semi_shared_va_a)

        # 新增：不相关模态在半公共空间的差异损失（核心改进）
        # A不应该在TV半公共空间有表示，通过跨空间差异实现
        loss += self.loss_diff(semi_shared_tv_t, semi_shared_ta_a)  # TV空间的T vs TA空间的A
        loss += self.loss_diff(semi_shared_tv_v, semi_shared_va_a)  # TV空间的V vs VA空间的A
        
        # V不应该在TA半公共空间有强表示
        loss += self.loss_diff(semi_shared_ta_t, semi_shared_tv_v)  # TA空间的T vs TV空间的V
        loss += self.loss_diff(semi_shared_ta_a, semi_shared_va_v)  # TA空间的A vs VA空间的V
        
        # T不应该在VA半公共空间有强表示
        loss += self.loss_diff(semi_shared_va_v, semi_shared_tv_t)  # VA空间的V vs TV空间的T
        loss += self.loss_diff(semi_shared_va_a, semi_shared_ta_t)  # VA空间的A vs TA空间的T

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss
