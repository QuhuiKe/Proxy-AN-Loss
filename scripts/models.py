import torch
import torch.nn as nn
import timm
from scripts.backbone import Resnet50, bn_inception
from scripts import loss
from scripts.eval_utils import l2_norm
import warnings
warnings.filterwarnings("ignore")


class Feature_Extractor(nn.Module):
    def __init__(self, embedding_size=128, n_mid=1024, pretrained=False, model='googlenet', is_norm=False, is_pooling=True, bn_freeze=True):
        super(Feature_Extractor, self).__init__()
        self.is_norm = is_norm
        self.is_pooling = is_pooling

        if model == 'BNInception':
            self.fc = nn.Sequential(
                nn.LayerNorm(n_mid),
                nn.BatchNorm1d(n_mid),
                nn.Linear(n_mid, embedding_size))
            self.backbone = bn_inception(pretrained=pretrained, Pool=is_pooling, bn_freeze=bn_freeze)
        elif model == 'Resnet50':
            self.fc = nn.Sequential(
                nn.LayerNorm(n_mid),
                nn.BatchNorm1d(n_mid),
                nn.Linear(n_mid, embedding_size))
            self.backbone = Resnet50(pretrained=pretrained, Pool=is_pooling, bn_freeze=bn_freeze)
        elif model == "DEIT":
            self.fc = nn.Sequential(
                nn.LayerNorm(n_mid),
                nn.BatchNorm1d(n_mid),
                nn.Linear(n_mid, embedding_size))
            self.backbone = timm.create_model("deit_small_distilled_patch16_224", pretrained=pretrained)
            self.backbone.reset_classifier(0, "token")
            nn.init.constant_(self.fc[2].bias.data, 0)
            nn.init.orthogonal_(self.fc[2].weight.data)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        embedding_y_orig = self.backbone(x)
        embedding_z = self.fc(embedding_y_orig)
        if self.is_norm:
            embedding_z = l2_norm(embedding_z)
        return embedding_z  


class Proxies(nn.Module):
    def __init__(self, nb_classes, sz_embed, loss="Proxy_AN"):
        torch.nn.Module.__init__(self)
        if loss == "Proxy_NCA":
            self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        else:
            self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self):
        P = self.proxies
        return P


class Metric_Model(nn.Module):
    def __init__(self, cfg, embedding_size=512, n_class=100, pretrained=False, is_norm=False, is_pooling=True, bn_freeze=True, alpha=32, mrg=0.1):
        super(Metric_Model, self).__init__()       
        self.is_pooling = is_pooling
        self.embedding_size = embedding_size
        backbone = cfg.MODEL.BACKBONE
        if backbone == 'Resnet50':
            n_mid = 2048
            self.feature_extractor = Feature_Extractor(self.embedding_size, n_mid, pretrained, model=backbone, is_norm=is_norm, is_pooling=is_pooling, bn_freeze=bn_freeze)
        elif backbone == 'DEIT':
            n_mid = 384
            self.feature_extractor = Feature_Extractor(self.embedding_size, n_mid, pretrained, model=backbone, is_norm=is_norm)
        elif backbone == 'BNInception':
            n_mid = 1024
            self.feature_extractor = Feature_Extractor(self.embedding_size, n_mid, pretrained, model=backbone, is_norm=is_norm)
        else:
            raise NotImplementedError
        
        self.cfg = cfg
        self.n_class = n_class
        self.clip = self.cfg.SOLVER.GRAD_CLIP
        self.loss = self.cfg.LOSS.METRIC_LOSS
        self.proxies = Proxies(n_class, embedding_size, self.loss)
        self.loss_fn = loss.Proxy_Anchor(n_class, embedding_size, alpha=alpha, mrg=mrg)


    def forward(self, x, t, opt_c=None, opt_pa=None):
        metrics = {}
        grad_clip = self.cfg.SOLVER.GRAD_L2_CLIP
        label = t.squeeze(-1)

        if opt_c is not None:
            opt_c.zero_grad()
        if opt_pa is not None:
            opt_pa.zero_grad()

        if opt_pa is not None:
            P = self.proxies()

        embedding_z = self.feature_extractor(x)

        jm = self.loss_fn(embedding_z, label, P, self.loss)
        metrics['J_m'] = jm.item()
        
        if opt_c is not None:
            jm.backward()
            if self.cfg.SOLVER.METRIC_TYPE == 'AdamW':
                if self.clip:
                    torch.nn.utils.clip_grad_value_(self.feature_extractor.parameters(), grad_clip) 
            opt_c.step()
        if opt_pa is not None:
            if self.clip:
                torch.nn.utils.clip_grad_value_(self.proxies.parameters(), grad_clip)
            opt_pa.step()

        return jm, metrics

