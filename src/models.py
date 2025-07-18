import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


class SelfAttention(nn.Module):
    """Self-attention module for feature enhancement"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, hidden_size]
        # 为了使用 MultiheadAttention，需要添加序列维度
        x_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        attn_output, _ = self.multihead_attn(x_seq, x_seq, x_seq)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_size]
        
        # 残差连接和层归一化
        output = self.layer_norm(x + self.dropout(attn_output))
        return output


class CrossSpaceAttention(nn.Module):
    """Cross-space attention module"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(CrossSpaceAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_space, context):
        # query_space: [batch_size, space_dim, hidden_size] - 某个空间的特征
        # context: [batch_size, total_dim, hidden_size] - 所有空间拼接的特征
        
        attn_output, _ = self.multihead_attn(query_space, context, context)
        
        # 残差连接和层归一化
        output = self.layer_norm(query_space + self.dropout(attn_output))
        return output


class GatedFusion(nn.Module):
    """Gated fusion module for multi-space features"""
    def __init__(self, hidden_size, num_spaces=3, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.num_spaces = num_spaces
        self.hidden_size = hidden_size
        
        # 门控网络：为每个空间学习权重
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(num_spaces)
        ])
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, space_features):
        # space_features: list of [batch_size, space_dim, hidden_size]
        # 每个元素代表一个空间的所有特征
        
        batch_size = space_features[0].size(0)
        
        # 计算每个空间的代表性特征（平均池化）
        space_representations = []
        for space_feat in space_features:
            # space_feat: [batch_size, space_dim, hidden_size]
            space_repr = torch.mean(space_feat, dim=1)  # [batch_size, hidden_size]
            space_representations.append(space_repr)
        
        # 计算门控权重
        gate_scores = []
        for i, space_repr in enumerate(space_representations):
            gate_score = self.gate_networks[i](space_repr)  # [batch_size, 1]
            gate_scores.append(gate_score)
        
        gate_scores = torch.cat(gate_scores, dim=1)  # [batch_size, num_spaces]
        gate_weights = self.softmax(gate_scores)  # [batch_size, num_spaces]
        
        # 加权融合
        weighted_features = []
        for i, space_feat in enumerate(space_features):
            # space_feat: [batch_size, space_dim, hidden_size]
            weight = gate_weights[:, i:i+1].unsqueeze(-1)  # [batch_size, 1, 1]
            weighted_feat = space_feat * weight  # 广播乘法
            weighted_features.append(weighted_feat)
        
        # 拼接所有加权特征
        final_features = torch.cat(weighted_features, dim=1)  # [batch_size, total_dim, hidden_size]
        
        return final_features


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        
        # 根据配置选择视觉特征处理方式
        if config.use_facet_visual:
            self.visual_size = config.facet_visual_size  # 47维Facet特征
        else:
            self.visual_size = config.visual_size  # 原始视觉特征
            
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.config.use_bert:
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        # 视觉特征处理：根据配置选择不同的处理方式
        if not config.use_facet_visual:
            # 原始方式：使用LSTM处理视觉特征
            self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
            self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
            self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        else:
            # 新方式：直接处理Facet特征，不使用LSTM
            self.visual_fc = nn.Sequential(
                nn.Linear(self.visual_size, config.hidden_size),
                self.activation,
                nn.LayerNorm(config.hidden_size)
            )
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        # 视觉投影层：根据特征处理方式调整
        if config.use_facet_visual:
            # Facet特征已经在visual_fc中处理过了，这里只需要恒等映射
            self.project_v = nn.Sequential()
            self.project_v.add_module('project_v', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.project_v.add_module('project_v_activation', self.activation)
            self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_v = nn.Sequential()
            self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
            self.project_v.add_module('project_v_activation', self.activation)
            self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # semi-shared encoders
        ##########################################
        # Text-Visual semi-shared encoder
        self.semi_shared_tv = nn.Sequential()
        self.semi_shared_tv.add_module('semi_shared_tv_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_shared_tv.add_module('semi_shared_tv_activation_1', nn.Sigmoid())
        
        # Text-Audio semi-shared encoder  
        self.semi_shared_ta = nn.Sequential()
        self.semi_shared_ta.add_module('semi_shared_ta_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_shared_ta.add_module('semi_shared_ta_activation_1', nn.Sigmoid())
        
        # Visual-Audio semi-shared encoder
        self.semi_shared_va = nn.Sequential()
        self.semi_shared_va.add_module('semi_shared_va_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_shared_va.add_module('semi_shared_va_activation_1', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))

        ##########################################
        # semi-shared discriminators
        ##########################################
        self.semi_discriminator_tv = nn.Sequential()
        self.semi_discriminator_tv.add_module('semi_discriminator_tv_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_discriminator_tv.add_module('semi_discriminator_tv_activation', self.activation)
        self.semi_discriminator_tv.add_module('semi_discriminator_tv_2', nn.Linear(in_features=config.hidden_size, out_features=3))

        self.semi_discriminator_ta = nn.Sequential()
        self.semi_discriminator_ta.add_module('semi_discriminator_ta_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_discriminator_ta.add_module('semi_discriminator_ta_activation', self.activation)
        self.semi_discriminator_ta.add_module('semi_discriminator_ta_2', nn.Linear(in_features=config.hidden_size, out_features=3))

        self.semi_discriminator_va = nn.Sequential()
        self.semi_discriminator_va.add_module('semi_discriminator_va_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.semi_discriminator_va.add_module('semi_discriminator_va_activation', self.activation)
        self.semi_discriminator_va.add_module('semi_discriminator_va_2', nn.Linear(in_features=config.hidden_size, out_features=3))

        ##########################################
        # 新的融合模块
        ##########################################
        # 获取融合模块参数
        fusion_num_heads = getattr(config, 'fusion_num_heads', 8)
        fusion_dropout = getattr(config, 'fusion_dropout', 0.1)
        use_residual = getattr(config, 'use_residual_fusion', True)
        
        # 自增强模块
        self.self_attn_pub = SelfAttention(config.hidden_size, 
                                          num_heads=fusion_num_heads, 
                                          dropout=fusion_dropout)
        self.self_attn_prv = SelfAttention(config.hidden_size, 
                                          num_heads=fusion_num_heads, 
                                          dropout=fusion_dropout)
        self.self_attn_semi = SelfAttention(config.hidden_size, 
                                           num_heads=fusion_num_heads, 
                                           dropout=fusion_dropout)
        
        # 跨空间注意力模块
        self.cross_attn_semi = CrossSpaceAttention(config.hidden_size, 
                                                  num_heads=fusion_num_heads, 
                                                  dropout=fusion_dropout)
        self.cross_attn_prv = CrossSpaceAttention(config.hidden_size, 
                                                 num_heads=fusion_num_heads, 
                                                 dropout=fusion_dropout)
        
        # 门控融合模块
        self.gated_fusion = GatedFusion(config.hidden_size, 
                                       num_spaces=3, 
                                       dropout=fusion_dropout)
        
        # 最终分类层
        self.final_classifier = nn.Sequential(
            nn.Linear(7 * config.hidden_size, config.hidden_size * 2),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(config.hidden_size * 2, output_size)
        )

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu() 
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        
        batch_size = lengths.size(0)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, 
                                        attention_mask=bert_sent_mask, 
                                        token_type_ids=bert_sent_type)      

            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            utterance_text = bert_output
        else:
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # 视觉特征处理：根据配置选择不同方式
        if self.config.use_facet_visual:
            # 使用Facet特征：需要正确处理维度
            if len(visual.shape) == 3:
                # visual shape: [seq_len, batch_size, feature_dim] 或 [batch_size, seq_len, feature_dim]
                # 需要确定visual的维度顺序
                
                if visual.shape[0] == batch_size:
                    # [batch_size, seq_len, feature_dim]
                    seq_len = visual.shape[1]
                    feature_dim = visual.shape[2]
                    
                    # 创建mask: [batch_size, seq_len, 1]
                    mask = (torch.arange(seq_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)).float().unsqueeze(2)
                    
                    # 计算加权平均
                    utterance_video = (visual * mask).sum(dim=1) / mask.sum(dim=1)
                    
                else:
                    # [seq_len, batch_size, feature_dim] - 需要转置
                    visual = visual.permute(1, 0, 2)  # 转为 [batch_size, seq_len, feature_dim]
                    seq_len = visual.shape[1]
                    feature_dim = visual.shape[2]
                    
                    # 创建mask: [batch_size, seq_len, 1]
                    mask = (torch.arange(seq_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)).float().unsqueeze(2)
                    
                    # 计算加权平均
                    utterance_video = (visual * mask).sum(dim=1) / mask.sum(dim=1)
                    
            elif len(visual.shape) == 2:
                # visual shape: [batch_size, feature_dim] - 已经是聚合后的特征
                utterance_video = visual
            else:
                raise ValueError(f"Unexpected visual shape: {visual.shape}")
            
            # 通过全连接层处理
            utterance_video = self.visual_fc(utterance_video)
            
        else:
            # 原始方式：使用LSTM提取视觉特征
            final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
            utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality (保持不变)
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        # 半公共空间判别器
        reversed_semi_tv_t = ReverseLayerF.apply(self.utt_semi_shared_tv_t, self.config.reverse_grad_weight)
        reversed_semi_tv_v = ReverseLayerF.apply(self.utt_semi_shared_tv_v, self.config.reverse_grad_weight)
        reversed_semi_tv_a = ReverseLayerF.apply(self.utt_semi_shared_ta_a, self.config.reverse_grad_weight)

        self.semi_domain_label_tv_t = self.semi_discriminator_tv(reversed_semi_tv_t)
        self.semi_domain_label_tv_v = self.semi_discriminator_tv(reversed_semi_tv_v)
        self.semi_domain_label_tv_a = self.semi_discriminator_tv(reversed_semi_tv_a)

        reversed_semi_ta_t = ReverseLayerF.apply(self.utt_semi_shared_ta_t, self.config.reverse_grad_weight)
        reversed_semi_ta_a = ReverseLayerF.apply(self.utt_semi_shared_ta_a, self.config.reverse_grad_weight)
        reversed_semi_ta_v = ReverseLayerF.apply(self.utt_semi_shared_tv_v, self.config.reverse_grad_weight)

        self.semi_domain_label_ta_t = self.semi_discriminator_ta(reversed_semi_ta_t)
        self.semi_domain_label_ta_a = self.semi_discriminator_ta(reversed_semi_ta_a)
        self.semi_domain_label_ta_v = self.semi_discriminator_ta(reversed_semi_ta_v)

        reversed_semi_va_v = ReverseLayerF.apply(self.utt_semi_shared_va_v, self.config.reverse_grad_weight)
        reversed_semi_va_a = ReverseLayerF.apply(self.utt_semi_shared_va_a, self.config.reverse_grad_weight)
        reversed_semi_va_t = ReverseLayerF.apply(self.utt_semi_shared_ta_t, self.config.reverse_grad_weight)

        self.semi_domain_label_va_v = self.semi_discriminator_va(reversed_semi_va_v)
        self.semi_domain_label_va_a = self.semi_discriminator_va(reversed_semi_va_a)
        self.semi_domain_label_va_t = self.semi_discriminator_va(reversed_semi_va_t)

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 新的融合策略
        o = self.advanced_fusion()
        return o
    
    def advanced_fusion(self):
        """新的多阶段融合策略"""
        batch_size = self.utt_private_t.size(0)
        
        # 阶段1: 准备三个空间的特征
        # Public space features (全共享空间)
        pub_features = torch.stack([
            self.utt_shared_t, 
            self.utt_shared_v, 
            self.utt_shared_a
        ], dim=1)  # [batch_size, 3, hidden_size]
        
        # Private space features (私有空间)
        prv_features = torch.stack([
            self.utt_private_t,
            self.utt_private_v, 
            self.utt_private_a
        ], dim=1)  # [batch_size, 3, hidden_size]
        
        # Semi-shared space features (半共享空间)
        semi_features = torch.stack([
            self.utt_semi_shared_tv_t,
            self.utt_semi_shared_tv_v,
            self.utt_semi_shared_ta_t, 
            self.utt_semi_shared_ta_a,
            self.utt_semi_shared_va_v,
            self.utt_semi_shared_va_a
        ], dim=1)  # [batch_size, 6, hidden_size]
        
        # 阶段2: 自增强 (Self-Enhancement)
        # 对每个空间内的特征进行自注意力增强
        pub_enhanced = []
        for i in range(pub_features.size(1)):
            enhanced = self.self_attn_pub(pub_features[:, i, :])
            pub_enhanced.append(enhanced)
        pub_enhanced = torch.stack(pub_enhanced, dim=1)  # [batch_size, 3, hidden_size]
        
        prv_enhanced = []
        for i in range(prv_features.size(1)):
            enhanced = self.self_attn_prv(prv_features[:, i, :])
            prv_enhanced.append(enhanced)
        prv_enhanced = torch.stack(prv_enhanced, dim=1)  # [batch_size, 3, hidden_size]
        
        semi_enhanced = []
        for i in range(semi_features.size(1)):
            enhanced = self.self_attn_semi(semi_features[:, i, :])
            semi_enhanced.append(enhanced)
        semi_enhanced = torch.stack(semi_enhanced, dim=1)  # [batch_size, 6, hidden_size]
        
        # 阶段3: 跨空间注意力 (Cross-Space Attention)
        # 拼接所有增强后的特征作为上下文
        all_context = torch.cat([pub_enhanced, prv_enhanced, semi_enhanced], dim=1)  # [batch_size, 12, hidden_size]
        
        # 对半共享和私有空间进行跨空间注意力
        semi_cross_attended = self.cross_attn_semi(semi_enhanced, all_context)  # [batch_size, 6, hidden_size]
        prv_cross_attended = self.cross_attn_prv(prv_enhanced, all_context)    # [batch_size, 3, hidden_size]
        
        # 阶段4: 门控融合 (Gated Fusion)
        space_features = [pub_enhanced, semi_cross_attended, prv_cross_attended]
        gated_features = self.gated_fusion(space_features)  # [batch_size, 12, hidden_size]
        
        # 阶段5: 最终分类
        # 根据你的设计，需要7个特征向量进行最终融合
        # 重新组织特征以得到7个向量
        final_seven_features = torch.cat([
            pub_enhanced.mean(dim=1),           # 1. 公共空间平均
            semi_cross_attended[:, 0, :],       # 2. TV半共享 - T
            semi_cross_attended[:, 1, :],       # 3. TV半共享 - V  
            semi_cross_attended[:, 2, :],       # 4. TA半共享 - T
            semi_cross_attended[:, 3, :],       # 5. TA半共享 - A
            semi_cross_attended[:, 4, :],       # 6. VA半共享 - V
            prv_cross_attended.mean(dim=1)      # 7. 私有空间平均
        ], dim=1)  # [batch_size, 7*hidden_size]
        
        output = self.final_classifier(final_seven_features)
        return output
    
    def reconstruct(self,):
        self.utt_t = (self.utt_private_t + self.utt_semi_shared_tv_t + self.utt_semi_shared_ta_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_semi_shared_tv_v + self.utt_semi_shared_va_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_semi_shared_ta_a + self.utt_semi_shared_va_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        # Semi-shared components
        self.utt_semi_shared_tv_t = self.semi_shared_tv(utterance_t)
        self.utt_semi_shared_tv_v = self.semi_shared_tv(utterance_v)
        
        self.utt_semi_shared_ta_t = self.semi_shared_ta(utterance_t)
        self.utt_semi_shared_ta_a = self.semi_shared_ta(utterance_a)
        
        self.utt_semi_shared_va_v = self.semi_shared_va(utterance_v)
        self.utt_semi_shared_va_a = self.semi_shared_va(utterance_a)

        # Fully shared components
        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
