""""
Define a generic GRM layer model
"""
from pickletools import decimalnl_short
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from libs.utils.graph_util import *
from graph.coco_data import *
from omegaconf import OmegaConf
from config.global_settings import GPU_ID
from torch.autograd import Variable
from torch.nn import Parameter
import math
from .init_weights import init_weights

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

cuda_suffix = 'cuda:' + str(GPU_ID) if len(str(GPU_ID)) == 1 else "cuda"
device = torch.device(cuda_suffix if torch.cuda.is_available() else "cpu")



class AttentionGraph(nn.Module):
    def __init__(self, model_dim=300, dropout=0.2):
        super(AttentionGraph, self).__init__()

        self.model_dim = model_dim
        self.linear_q = nn.Linear(model_dim,model_dim)
        self.linear_k = nn.Linear(model_dim,model_dim)
        self.linear_v = nn.Linear(model_dim,model_dim)

        self.linear_o = nn.Linear(model_dim,model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, node):

        node = node.cuda()
        q = self.linear_q(node)
        
        k = self.linear_k(node) 
        v = self.linear_v(node)

        #attention = torch.mm(q, k.t())
        attention = torch.mm(q, k.t()) / math.sqrt(self.model_dim)
        node_att = F.softmax(attention, dim=-1)

        node1 = torch.mm(node_att, v)
        node1 = torch.mean(node1,dim=0,keepdim=True)
        
        node2= self.linear_o(node1)
        node2 = self.dropout(node2)
        output = node + node2
    
        return output

#Graph convolution
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight.cuda())
        #support = torch.matmul(input, self.weight)
        adj=adj.to(input.cuda())
        #adj=adj.to(input.cpu())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GRModule(nn.Module):
    def __init__(self, x_channal=256, M=16):
        super(GRModule, self).__init__()

        self.M = M

        self.phi_conv = nn.Conv2d(x_channal, M, kernel_size=1, stride=1, padding=0, bias=True)

        self.glob_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.glob_conv = nn.Conv2d(x_channal, M, kernel_size=1, stride=1, padding=0,bias=False)


        self.out_conv = nn.Conv2d(x_channal, x_channal, kernel_size=1, stride=1, padding=0, bias = False)
        
        self.bn = BatchNorm2d(x_channal)

    def forward(self, x):
        x_phi_conv = self.phi_conv(x)
        #x_phi_conv = x
        x_phi = x_phi_conv.view([x_phi_conv.shape[0], -1, self.M])
        x_phi = F.relu(x_phi)#[B, 1681, 64]

        x_phi_T = x_phi_conv.view([x_phi_conv.shape[0], self.M, -1])
        x_phi_T = F.relu(x_phi_T)#[B, 64, 1681]

        x_glob_pool = self.glob_pool(x)
        x_glob_conv = self.glob_conv(x_glob_pool)
        
        x_glob_diag = torch.zeros(x_glob_conv.shape[0], self.M, self.M).cuda()

        for i in range(x_glob_conv.shape[0]):
            x_glob_diag[i, :, :] = torch.diag(x_glob_conv[i, :, :, :].reshape(1,self.M))#[B, 64, 64]

        x_glob_diag = torch.sigmoid(x_glob_diag)
        A = torch.matmul(torch.matmul(x_phi, x_glob_diag), x_phi_T)#[B, 1681, 1681]
        A = F.softmax(A, dim=-1)
    
        D_sqrt = torch.zeros_like(A).cuda()

        diag_sum =torch.sum(A, 2)
        
        for i in range(diag_sum.shape[0]):
            diag_sqrt = 1.0 / torch.sqrt(diag_sum[i, :])
            diag_sqrt[torch.isnan(diag_sqrt)] = 0
            diag_sqrt[torch.isinf(diag_sqrt)] = 0

            D_sqrt[i, :, :] = torch.diag(diag_sqrt)
        
        I = torch.eye(D_sqrt.shape[1]).cuda()
        I = I.repeat(D_sqrt.shape[0], 1, 1)

        L = I - torch.matmul(torch.matmul(D_sqrt, A), D_sqrt)
     
        output = torch.matmul(L, x.reshape(x.shape[0], -1, x.shape[1]))
        
        return output
       





class GraphLayer(nn.Module):
    #def __init__(self, num_state, num_node, num_class):
    def __init__(self, inp, vis_node):
        super().__init__()
        #num_state=256
        self.inp = inp.transpose(2,1)
        self.vis_node = vis_node.transpose(2,1)
        B,num_state,num_node = self.vis_node.size()
        B,self.in_channels,num_class = self.inp.size()
        #print("#########num_node:",num_node)
        self.vis_gcn = GCN(num_state, num_node)
        self.word_gcn = GCN(num_state, num_class)
        self.transfer = GraphTransfer(num_state)
       
        self.dropout = nn.Dropout(0.1)

    def forward(self):
        inp = self.word_gcn(self.inp)
        new_V = self.vis_gcn(self.vis_node)
        class_node, vis_out = self.transfer(inp, new_V)

        class_node = inp + class_node
        new_V = vis_out + new_V
        
        return class_node, new_V

class GCN(nn.Module):
    def __init__(self, num_state=256, num_node=20, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(
            num_node,
            num_node,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            num_state,
            num_state,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class GraphTransfer(nn.Module):
    def __init__(self, in_dim):
        super(GraphTransfer, self).__init__()
        self.channle_in = in_dim
        self.query_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv_vis = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_word = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax_vis = nn.Softmax(dim=-1)
        self.softmax_word = nn.Softmax(dim=-2)

    def forward(self, word, vis_node):
        m_batchsize, C, Nc = word.size()
        m_batchsize, C, Nn = vis_node.size()

        proj_query = self.query_conv(word).view(m_batchsize, -1, Nc).permute(0, 2, 1)
        proj_key = self.key_conv(vis_node).view(m_batchsize, -1, Nn)

        energy = torch.bmm(proj_query, proj_key)
        attention_vis = self.softmax_vis(energy).permute(0, 2, 1)
        attention_word = self.softmax_word(energy)

        proj_value_vis = self.value_conv_vis(vis_node).view(m_batchsize, -1, Nn)
        proj_value_word = self.value_conv_word(word).view(m_batchsize, -1, Nc)

        class_out = torch.bmm(proj_value_vis, attention_vis)
        node_out = torch.bmm(proj_value_word, attention_word)
        return class_out, node_out


#Semantic Mapping Module
class SemanticToLocal(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels):
        super(SemanticToLocal, self).__init__()

        # It is necessary to calculate the mapping weight matrix from 
        #symbol nodes to local features for each image. [?, H*W, M]
        # The W in the paper is as follows
        self.conv1 = nn.Conv2d(256 +256, 1,
                              kernel_size=1, stride=1)
        
        self.relu = nn.ReLU(inplace=False)

    def compute_compat_batch(self, batch_input, batch_evolve):
        # batch_input [H, W, Dl]
        # batch_evolve [M, Dc]
        # [H, W, Dl] => [H * W, Dl] => [H*W, M, Dl]
        H = batch_input.shape[0]
        W = batch_input.shape[1]
        M = batch_evolve.shape[0]
        Dl = batch_input.shape[-1]
        batch_input = batch_input.reshape( H * W, Dl)
        batch_input = batch_input.unsqueeze(1).repeat([1,M,1])
        # [M,Dc] => [H*W, M, Dc]
        batch_evolve = batch_evolve.unsqueeze(0).repeat([H*W, 1, 1])
        # [H*W, M, Dc+Dl] 
        batch_concat = torch.cat([batch_input, batch_evolve], axis=-1)
        # [H*W, M, Dc+Dl] =>[1,H*W, M, Dc+Dl]
        batch_concat = batch_concat[np.newaxis,:,:,:]
        # [H*W, M, Dc+Dl] =>[1,Dc+Dl,H*W, M]
        batch_concat = batch_concat.transpose(2,3).transpose(1,2)
        #[1,Dc+Dl,H*W, M] =>[1,1,H*W, M]
        mapping = self.conv1(batch_concat)
        #[1,1,H*W, M] => [1, H*W, M, 1]
        mapping = mapping.transpose(1,2).transpose(2,3)
        #[1,1,H*W, M] => [H*W, M, 1]
        mapping = mapping.view(-1,mapping.size(2),mapping.size(3))
        #[H*W, M,1] => [H*W, M]
        mapping = mapping.view(mapping.size(0), -1)
        mapping = F.softmax(mapping, dim=0)
        return  mapping

    def forward(self, x, evolved_feat):
        # [?, Dl, H, W] , [?, M, Dc]
        input_feat = x 
        evolved_feat = evolved_feat
        # [?, H, W, Dl]
        input_feat = input_feat.transpose(1,2).transpose(2, 3)
        batch_list = []
        for index in range(input_feat.size(0)):
            batch = self.compute_compat_batch(input_feat[index], evolved_feat[index])
            batch_list.append(batch)
        # [?, H*W, M]
        mapping = torch.stack(batch_list, dim=0)
        # [?, M, Dc] => [? * M, Dc] => [? * M, Dl] => [?, M, Dl]
        Dl = input_feat.size(-1)
        M = evolved_feat.size(1)
        H = input_feat.size(1)
        W = input_feat.size(2)
        # [?, M, Dc] => [? * M, Dc]
        #[?, H*W, M] @ [? , M, Dl] => [?, H*W, Dl]
        applied_mapping = torch.bmm(mapping, evolved_feat)
        applied_mapping = self.relu(applied_mapping)
        #[?, H*W, Dl] => [?, H, W, Dl]
        applied_mapping = applied_mapping.reshape(input_feat.size(0), H , W, Dl)
        #[?, H, W, Dl] => [?, Dl, H, W]
        applied_mapping = applied_mapping.transpose(2,3).transpose(1,2)

        return applied_mapping

#overall model layer
class CDGR(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(CDGR, self).__init__()

        self.node_atte = AttentionGraph(300)
        self.sprial = GRModule(256, 16)

        self.graph_reasoning1 = GraphConvolution(300,256)
        self.graph_reasoning2 = GraphConvolution(256,256)

        self.graph_weight = nn.Conv2d(input_feature_channels, input_feature_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.final = nn.Sequential(nn.Conv2d(256* 2, 256, kernel_size=1, bias=False))
                                     #BatchNorm2d(input_feature_channels))

        self.semantic_to_local = SemanticToLocal(input_feature_channels, visual_feature_channels)
        
        self.graph_adj_mat = torch.FloatTensor(graph_adj_mat).cuda()
        self.visual_feature_channels = visual_feature_channels
        self.fasttest_embeddings = torch.FloatTensor(fasttest_embeddings).cuda()
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        #[？，M, H*W]
        #x = self.conv1(x)
        visual_feat = x

        fasttest_embeddings = self.node_atte(self.fasttest_embeddings)
        #fasttest_embeddings = self.fasttest_embeddings.unsqueeze(0)
        fasttest_embeddings = fasttest_embeddings.repeat(visual_feat.size(0), 1, 1).to(visual_feat.cuda())
        graph_norm_adj = normalize_adjacency(self.graph_adj_mat)
        
        batch_list = []
        for index in range(visual_feat.size(0)):
            batch = self.graph_reasoning1(fasttest_embeddings[index], graph_norm_adj)
            batch = F.relu(batch)
            batch_list.append(batch)
        # [?, M, H*W]
        evolved_feat = torch.stack(batch_list, dim=0)
        batch_list1 = []
        for index in range(evolved_feat.size(0)):
            evolved_feats = F.dropout(evolved_feat[index], 0.3)
            batch1 = self.graph_reasoning2(evolved_feats, graph_norm_adj)
            batch1 = F.relu(batch1)
            batch_list1.append(batch1)
        # [?, M, H*W]
        evolved_feat1 = torch.stack(batch_list1, dim=0)

        out2 = self.sprial(x)

        out1, out2 = GraphLayer(evolved_feat1,out2).cuda().forward()
        out1 = torch.transpose(out1, 2, 1)
        enhanced_feat = self.semantic_to_local(x, out1)
        out1 = enhanced_feat
        
        out2 = out2.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        out2 = self.graph_weight(out2)
        out2 = F.relu(out2)
    
        out = self.final(torch.cat((out1, out2), 1))

        out = out + x

        return out


if __name__ == "__main__":
   pass