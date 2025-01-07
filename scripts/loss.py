import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.eval_utils import l2_norm


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


class Proxy_Anchor(nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.all_classes = torch.arange(self.nb_classes)
        
    def forward(self, X, label, P, loss="Proxy_AN"):
        if loss == "Proxy_NCA":
            batch       = 3*torch.nn.functional.normalize(X, dim=1)
            PROXIES     = 3*torch.nn.functional.normalize(P, dim=1)
            pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in label])
            neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in label])
            neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])
            dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies).pow(2),dim=-1)
            dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies).pow(2),dim=-1)
            loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
            return loss  
        
        else:
            T = label
            cos = F.linear(l2_norm(X), l2_norm(P))  
            P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
            N_one_hot = 1 - P_one_hot

            pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
            neg_exp = torch.exp(self.alpha * (cos + self.mrg))

            with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   
            num_valid_proxies = len(with_pos_proxies)   
            
            if loss == "Proxy_AN":
                P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
                pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies

                N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=1)
                neg_term = torch.log(1 + N_sim_sum).sum() / X.shape[0]
            
            elif loss == "Proxy_anchor":
                P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
                pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies

                N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
                neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

            loss = pos_term + neg_term
            return loss
    