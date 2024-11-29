import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from layers.xtansformer import XTransformerDecoder
from manifolds.lorentz import Lorentz
import manifolds.lmath as lmath
import manifolds.hierpe as hpe
import math
class LLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True, temp=50, riemannian = False):
        super(LLinear, self).__init__()
        self.riemannian = lmath.RiemannianGradient
        self.riemannian.c = c
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.zeros(out_features, in_features,dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.lorentz = Lorentz(k=c, learnable=True)
        self.margin = torch.ones(1) * 0.1
        self.temp = temp
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        # nn.init.kaiming_normal_(self.bias)
        if self.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias,-bound, bound)

    def forward(self, x, c=None):
        
        self.lorentz.clamp_k()
        c = self.lorentz.k.exp()
        
        mv = x@self.weight
        # if self.bias is None:
        return lmath.expmap0(mv, k=c)
        # else:
        #     bias = lmath.expmap0(self.bias, k=c)
        #     return self.lorentz.projx(mv+bias)
    def tree_triplet_loss(self, node, depth):
        # self.margin.cuda()
        loss = 0
        count = 0
        for i in range(depth-2):
            child_node = node[i]
            lca_node = node[i+1]
            lca_node2 = node[i+2]
            # lca2_node = node/ㄴ[i+2]
            B, N,_= child_node.shape
            # N, _ = lca2_node.shape()
            # for i in range(4):
            #     loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,i::4],lca_node[:,(i//2)::2]) - self.lorentz.dist_n(child_node[:,i::4],lca_node2)+self.margin))*(N/4)
                # loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,(i+2)::4],lca_node[:,((i+2)//2)::2]) -self.lorentz.dist_n(child_node[:,(i+2)::4],lca_node[:,(i//2)::2])+self.margin))*(N/4)
            for i in range(2):
                loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,i::4],lca_node2) - self.lorentz.dist_n(child_node[:,i::4],lca_node[:,((i+2)//2)::2])+0.1))*(N/4)
                loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,(i+2)::4],lca_node2) -self.lorentz.dist_n(child_node[:,(i+2)::4],lca_node[:,(i//2)::2])+0.1))*(N/4)
            # for i in range(4):
            #     loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,i::4],lca_node2)))*(N*0.1/4)
            count += N
            
        child_node = node[-2]
        lca_node = node[-1]
        
        for i in range(2):
        # # loss += F.relu(torch.mean(self.lorentz.dist_n(lca_node[:,1], ori_node)+self.lorentz.dist_n(lca_node[:,0], ori_node)-2*self.lorentz.dist_n(lca_node[:,0], lca_node[:,1]))+self.margin)*2
            loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,i::4],lca_node[:,(i//2)::2])-self.lorentz.dist_n(child_node[:,i::4], lca_node[:,((i+2)//2)::2])+0.1))
            loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,(i+2)::4], lca_node[:,((i+2)//2)::2])-self.lorentz.dist_n(child_node[:,(i+2)::4],lca_node[:,(i//2)::2])+0.1))
        
        # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,0],lca_node[:,0]) - self.lorentz.dist_n(child_node[:,0],ori_node))+self.margin)
        # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,1],lca_node[:,0]) - self.lorentz.dist_n(child_node[:,1],ori_node))+self.margin)
        # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,2],lca_node[:,1]) - self.lorentz.dist_n(child_node[:,2],ori_node))+self.margin)
        # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,3],lca_node[:,1]) - self.lorentz.dist_n(child_node[:,3],ori_node))+self.margin)
        
        # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,2],lca_node[:,1]) - self.lorentz.dist_n(child_node[:,2],lca_node[:,0]))+self.margin)
        # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,3],lca_node[:,1]) - self.lorentz.dist_n(child_node[:,3],lca_node[:,0]))+self.margin)
        # print("tree : {}".format(loss))
        # loss = F.relu(loss + self.margin)
        return loss
    def reg_loss(self, node, depth):
        loss = 0
        count = 0
        for i in range(depth-1):
            child_node = node[i]
            lca_node = node[i+1]
            B, N, _ =child_node.shape
            loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,0::2],lca_node)))*(N/2)
            loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,1::2],lca_node)))*(N/2)
            count += N
        lca_node = node[-2]
        ori_node = node[-1].squeeze(1)
        count += 2
        loss += F.relu(torch.mean(self.lorentz.dist_nn(lca_node[:,1], ori_node)+self.lorentz.dist_nn(lca_node[:,0], ori_node)))
        # loss = torch.log(loss)
        # print("reg_loss : {}".format(loss/count))
        return loss/count
    def node_loss(self, node, depth):
        loss = 0
        # for i in range(depth-1):
        #     child_node = node[i]
        #     lca_node = torch.repeat_interleave(node[i+1], 2, dim=1)
        self.lorentz.clamp_k()
        c = self.lorentz.k.exp()
            
            
        # for i in range(depth-1):
        #     child_node = node[i]
        #     lca_node = node[i+1]
        #     B, N, _ =lca_node.shape
        #     loss += torch.mean(F.relu(self.lorentz.dist_n(child_node[:,0::2],child_node[:,1::2])+self.margin))*N/2
            # loss += F.relu(torch.mean(self.lorentz.dist_nn(child_node[:,1::2],lca_node) - self.lorentz.dist_nn(child_node[:,0::2],child_node[:,1::2]))+self.margin)*N/2
            # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,1::2],lca_node) - self.lorentz.dist_n(child_node[:,0::2],child_node[:,1::2]))+self.margin)*N
        # for i in range(depth-1):
        #     child_node = node[i]
        #     lca_node = node[i+1]
        #     B, N, _ =lca_node.shape
        #     loss += F.relu(torch.mean(self.lorentz.dist_nn(child_node[:,0::2],lca_node) - self.lorentz.dist_nn(child_node[:,0::2],child_node[:,1::2]))+self.margin)*N/2
        #     loss += F.relu(torch.mean(self.lorentz.dist_nn(child_node[:,1::2],lca_node) - self.lorentz.dist_nn(child_node[:,0::2],child_node[:,1::2]))+self.margin)*N/2
        #     # loss += F.relu(torch.mean(self.lorentz.dist_n(child_node[:,1::2],lca_node) - self.lorentz.dist_n(child_node[:,0::2],child_node[:,1::2]))+self.margin)*N
        # lca_node = node[-2]
        # ori_node = node[-1].squeeze(1)
        
        # # # loss += F.relu(torch.mean(self.lorentz.dist_n(lca_node[:,1], ori_node)+self.lorentz.dist_n(lca_node[:,0], ori_node)-2*self.lorentz.dist_n(lca_node[:,0], lca_node[:,1]))+self.margin)*2
        # loss += F.relu(torch.mean(self.lorentz.dist_nn(lca_node[:,1], ori_node) - self.lorentz.dist_nn(lca_node[:,0], lca_node[:,1]))+self.margin)
        # loss += F.relu(torch.mean(self.lorentz.dist_nn(lca_node[:,0], ori_node) - self.lorentz.dist_nn(lca_node[:,0], lca_node[:,1]))+self.margin)
        # loss += F.relu(torch.mean(self.lorentz.dist_n(lca_node[:,0], ori_node)-self.lorentz.dist_n(lca_node[:,0], lca_node[:,1]))+self.margin)
        for i in range(depth-1):
            child_node = node[i]
            lca_node = torch.repeat_interleave(node[i+1], 2, dim=1)
            
            # child_node = F.normalize(child_node, dim = -1)
            # lca_node = F.normalize(lca_node, dim = -1)            
            cn_time = torch.sqrt(1/c + torch.sum(child_node*child_node, dim = -1))
            lca_time = torch.sqrt(1/c + torch.sum(lca_node*lca_node, dim = -1))
            
            B, N, _ = lca_node.shape
            logits_time = torch.einsum('bn, bm -> bnm', cn_time, cn_time)
            logits_true_time = torch.einsum('bn, bm -> bnm', cn_time, lca_time)
            # logits_time = cn_time @ cn_time.transpose(-2, -1)
            # logits_true_time = lca_time @ lca_time.transpose(-2, -1)
            
            logits = child_node @ child_node.transpose(-2, -1)
            logits_true = child_node @ lca_node.transpose(-2, -1)
            
            logits = logits - logits_time
            logits_true = logits_true - logits_true_time
            
            eyemask = torch.eye(N)
            eyemask = eyemask.cuda()
            n_eyemask = torch.ones_like(eyemask) - eyemask
            target = torch.arange(N).cuda()
            target = target.expand(B, -1)
            
            logits = (logits * n_eyemask + logits_true * eyemask)
            logits = -c*logits
            logits = logits.clamp(min = 1+1e-8)
            logits = torch.acosh(logits)/(c**0.5)
            loss += F.cross_entropy(logits, target)
        # # chile_node = node[-3]
        
        # child_node = node[-2]
        loss = loss
        # print("node : {}".format(loss))
        return loss
    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )
    
    def reconstruction_loss(self, node, visual):
        loss = 0
        # query = visual[:,0]
        query = visual

        # act = torch.einsum('bmc, bc -> bm',visual[:, 1:], query)
        # act_min, _ = torch.min(act, dim=-1, keepdim=True)
        # act_max, _ = torch.max(act, dim=-1, keepdim=True)
        # act = 1 - ((act-act_min)/(act_max - act_min+1e-5))
        # neg_query = torch.mean(visual[:,1:]*act[...,None] ,dim=(-2, -1))
        # pos_sample = torch.mean(node, dim=1)
        pos_sample = node[:,-1]
        # neg_sample = node[:, 1]
        # loss = 2 - torch.mean(F.cosine_similarity(query, pos_sample) - F.cosine_similarity(query, neg_sample)) - torch.mean(F.cosine_similarity(neg_query[...,None], neg_sample))
        # loss += torch.log(2 - torch.mean(F.cosine_similarity(query, pos_sample.squeeze(1))))
        loss = 1 - torch.mean(F.cosine_similarity(query, pos_sample.squeeze(1)))
        # loss = torch.log(2 - torsch.mean(F.cosine_similarity(query, pos_ssample)))

        # loss = torch.log(F.relu(torch.mean(self.lorentz.dist_n(query,pos_sample))))
        # loss = F.relu(torch.mean(self.lorentz.dist_n(query.detach(),pos_sample) - self.lorentz.dist_n(query,neg_sample))+0.5)
        print("recon : {}".format(loss))
        # loss =  F.relu(loss)
        return loss

class HyboNet(torch.nn.Module):
    def __init__(self, c, bdim, dim, depth, temp):
        super(HyboNet, self).__init__()
        # self.manifold = Lorentz(max_norm=max_norm)

        # self.emb_entity = ManifoldParameter(self.manifold.random_normal((len(d.entities), dim), std=1./math.sqrt(dim)), manifold=self.manifold)
        # self.relation_bias = nn.Parameter(torch.zeros((len(d.relations), dim)))
        # self.relation_transform = nn.Parameter(torch.empty(len(d.relations), dim, dim))
        # nn.init.kaiming_uniform_(self.relation_transform)
        # self.scale = nn.Parameter(torch.ones(()) * max_scale, requires_grad=False)
        # self.margin = margin
        # self.bias_head = torch.nn.Parameter(torch.zeros(len(d.entities)))
        # self.bias_tail = torch.nn.Parameter(torch.zeros(len(d.entities)))
        # self.loss = torch.nn.BCEWithLogitsLoss()

        self.Etransformer = XTransformerDecoder(visual_dim=bdim)
        self.Llinear = LLinear(dim, bdim, c, temp)
        repeat = 2**(depth-1)
        self.node_mean =1.5*(torch.randn(2**(depth), bdim, dtype=torch.float))
        self.node_mean = nn.Parameter(torch.repeat_interleave(self.node_mean, 2, dim =0) + 0.01*torch.randn(2**(depth+1), bdim, dtype=torch.float))
        self.node_log_sigma = nn.Parameter(torch.randn(2**(depth+1), bdim, dtype=torch.float))
        nn.init.xavier_uniform_(self.node_log_sigma)
        # nn.init.kaiming_uniform_(tmp)
        # tmp = torch.repeat_interleave(tmp, repeat, dim=0)
        # for i in range(4):
        #     e = torch.randn(repeat, bdim) / math.sqrt(bdim) * 0.9
        #     tmp[i*repeat:(i+1)*repeat] = tmp[i*repeat:(i+1)*repeat] + e
        # self.node_set = nn.Parameter(tmp)
        
        # tmp = torch.zeros(2**(depth+1)-1, dim).cuda()
        # nn.init.kaiming_normal_(tmp)
        #  = tmp
        # self.node_set = self.node_embedding(bdim, depth)
        self.depth = depth
        # self.hpe = hpe.HierarchicalPE(bdim, depth)
    def node_embedding(self, dim, depth):
        # node_set = []
        # for i in range(depth):````````````````
        #     tmp = nn.Parameter(torch.zeros(2**(depth-i), dim)).cuda()
        #     nn.init.kaiming_normal_(tmp)
        #     tmp = tmp.to(torch.float16)
        #     node_set.append(tmp)
        repeat = 2**(depth-1)
        tmp =torch.zeros(4, dim)
        nn.init.kaiming_uniform_(tmp)
        tmp = torch.repeat_interleave(tmp, repeat, dim=0)
        for i in range(4):
            e = torch.randn(repeat, dim) / math.sqrt(dim) * 0.9
            tmp[i*repeat:(i+1)*repeat] = tmp[i*repeat:(i+1)*repeat] + e
        tmp = nn.Parameter(tmp).cuda()
        
        # tmp = torch.zeros(2**(depth+1)-1, dim).cuda()
        # nn.init.kaiming_normal_(tmp)
        node_set = tmp.to(torch.float16)
        return node_set
    def forward(self, visual, skip_head):
        # visual = visual.permute(0, 2, 1)
        B, M, C = visual.shape
        # self.node_set.detach_()
        # visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)
        # N, _ = self.node_set.shape
        node_memory = []
        node_memoryH = []
        
        ####
        # node = 10*self.node_set + 0*self.hpe(self.node_set, self.depth).expand(B, -1, -1)
        # node = 100*(self.node_mean.expand(B,-1,-1) + self.node_log_sigma.exp().expand(B, -1, -1)*torch.randn_like(self.node_log_sigma, device=self.node_log_sigma.device, dtype=self.node_log_sigma.dtype))
        # node = node.type(visual.type())
        # visual.detach_()
        if C < 500:
            E_n, kl = self.Etransformer(self.node_mean, self.node_log_sigma, visual[:,1:])
        else:
            E_n, kl = self.Etransformer(self.node_mean, self.node_log_sigma, visual)
        H_n = self.Llinear(E_n[0])
        for i in range(self.depth):
            start = 2**(self.depth+2)- 2**(self.depth+2 - i)
            end = 2**(self.depth+2) - 2**(self.depth+1-i)
            if skip_head:
                node_memory.append(E_n[0][:,start:end])
            else:
                node_memoryH.append(H_n[:,start:end])
                node_memory.append(E_n[0][:,start:end])
            # E_n = self.Etransformer(node+temp, visual)
            

            # if i == self.depth:
            #     E_n = self.Etransformer(node+temp, visual)
            #     H_n = self.Llinear(E_n)
            # else:
            #     E_n = self.Etransformer(node+temp, visual)
            #     H_n = self.Llinear(E_n)
            #     node = (node[:,0::2] + node[:,1::2])/2
            #     temp = F.normalize(E_n[:,0::2] + E_n[:,1::2], dim=-1)
            # if skip_head:
            #     node_memory.append(E_n)
            # else:
            #     node_memory.append(H_n)
            
        if skip_head:
            return [node_memory,E_n[0], E_n[1]]
        else:
            # recon_loss = self.Llinear.reconstruction_loss(E_n, visual[:,0])
            # recon_loss = 0
            tree_loss = self.Llinear.tree_triplet_loss(node_memoryH, self.depth)
            node_loss = self.Llinear.node_loss(node_memoryH, self.depth)
            loss = 0
            # reg_loss= self.Llinear.reg_loss(node_memory, self.depth)
            print("tree : {}, node : {}".format(tree_loss, node_loss))
            # loss = [tree_loss, node_loss]
            # loss += tree_loss.sum()/100
            loss += node_loss.sum()/10
            loss += kl/100
            return [node_memory, loss, E_n[0]]
