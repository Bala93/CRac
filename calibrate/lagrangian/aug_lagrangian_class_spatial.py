import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from torch import distributed as dist
import torch
import torch.nn.functional as F

from .penalty import get_penalty_func
from .aug_lagrangian import AugLagrangian
from kornia.morphology import dilation
from skimage.segmentation import find_boundaries
# from ..utils import reduce_tensor

logger = logging.getLogger(__name__)


class SpatialAugLagrangianClass(AugLagrangian):
    def __init__(
        self,
        num_classes: int = 10,
        margin: float = 10,
        penalty: str = "p2",
        lambd_min: float = 1e-6,
        lambd_max: float = 1e6,
        lambd_step: int = 1,
        rho_min: float = 1,
        rho_max: float = 10,
        # rho_update: bool = False,
        rho_step: int = -1,
        gamma: float = 1.2,
        tao: float = 0.9,
        normalize: bool = True,
        is_softmax : bool = False,
        is_foreground : bool = False,
        constr_type: str = "abs",
        is_undersample: bool = False
    ):
        assert penalty in ("p2", "p3", "phr", "relu"), f"invalid penalty: {penalty}"
        self.num_classes = num_classes
        self.margin = margin
        self.penalty_func = get_penalty_func(penalty)
        self.lambd_min = lambd_min
        self.lambd_max = lambd_max
        self.lambd_step = lambd_step
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_update = rho_step > 0
        self.rho_step = rho_step
        self.gamma = gamma
        self.tao = tao
        self.normalize = normalize
        self.is_softmax = is_softmax
        self.is_foreground = is_foreground
        self.constr_type = constr_type
        self.is_undersample = is_undersample

        self.lambd = self.lambd_min * torch.ones(self.num_classes, requires_grad=False).cuda()
        # self.rho = self.rho_min
        # class-wise rho
        self.rho = self.rho_min * torch.ones(self.num_classes, requires_grad=False).cuda()
        # for updating rho
        self.prev_constraints, self.curr_constraints = None, None

    def get_constraints(self, logits, targets, dim: int = -1):
        
        targets = targets.unsqueeze(1)
    
        unfold = torch.nn.Unfold(kernel_size=(3, 3),padding=3 // 2)    
        rmask = []
        
        utarget = unfold(targets.float())
        bs, _, h, w = targets.shape
            
        for ii in range(self.num_classes):
            rmask.append(torch.sum(utarget == ii,1)/3**2)

        rmask = torch.stack(rmask,dim=1)
        rmask = rmask.reshape(bs, self.num_classes, h, w)
        
        # constraints = torch.abs(rmask - logits) ## without margin
        # constraints = F.relu(torch.abs(rmask - logits) - self.margin)
        # constraints = F.relu(logits - self.margin * rmask)
        
        abs_constr = lambda pr, lg : torch.abs(pr - lg)
        max_constr = lambda pr, lg : F.relu(lg - pr)
        
        if self.is_softmax:
            logits = F.softmax(logits, dim=1)
        
        if self.constr_type == 'abs':    
            constraints = abs_constr(self.margin * rmask, logits)
        else:    
            constraints = max_constr(self.margin * rmask, logits)

        # max_values = logits.amax(dim=dim, keepdim=True)
        # diff = max_values - logits
        # constraints = diff - self.margin
        # if self.normalize:
        #     constraints = constraints / max_values
        return constraints
    
    def get(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h = self.get_constraints(logits, targets)
        
        # logits = logits.movedim(1, -1)  # move class dimension to last
        h = h.movedim(1, -1) ## to have same dimension as that of logit.
        p, _ = self.penalty_func(h, self.lambd, self.rho)
        # penalty = p.sum(dim=-1).mean()  # sum over classes and average over samples (and possibly pixels)
        penalty = p.mean()
        constraint = h.mean()
        return penalty, constraint
    
    def reset_update_lambd(self, epoch):
        # if (epoch + 1) % self.lambd_step == 0:
        self.grad_p_sum = torch.zeros_like(self.lambd)
        self.sample_num = 0
        self.curr_constraints = torch.zeros_like(self.rho)
        self.class_weights = torch.zeros_like(self.lambd)

    def update_lambd(self, logits, targets, epoch):
        """update lamdb based on the gradeint on the logits
        """
        
        h = self.get_constraints(logits, targets)
            
        # if (epoch + 1) % self.lambd_step == 0:
        # logits = logits.movedim(1, -1)  # move class dimension to last
        h = h.movedim(1, -1) ## to have same dimension as that of logit.
        _, grad_p = self.penalty_func(h, self.lambd, self.rho)
        grad_p = torch.clamp(grad_p, min=self.lambd_min, max=self.lambd_max)
        
        '''
        Better to skip the elements which doesn't come in the foreground dilated, 
        make this a 1d vector. 
        '''
        grad_p = grad_p.flatten(start_dim=0, end_dim=-2)
        h = h.flatten(start_dim=0, end_dim=-2)
        targets = targets.flatten(start_dim=0, end_dim=-1) 

        if self.is_foreground:
            fg_idx = targets > 0
            fg_idx = fg_idx.unsqueeze(1)
            kernel = torch.ones(3,3, device=fg_idx.device)
            fg_idx = dilation(fg_idx, kernel).bool()
            fg_idx = fg_idx.flatten(start_dim=0, end_dim=-1) 
            targets = targets[fg_idx]
            grad_p = grad_p[fg_idx,:]
            h = h[fg_idx,:]
            
        if self.is_boundary:
            bnd = find_boundaries(targets)
            targets = targets[fg_idx]
            grad_p = grad_p[fg_idx,:]
            h = h[fg_idx,:]            
            
        if self.is_undersample:
            _, cnt = torch.unique(targets, return_counts=True)
            mcnt = min(cnt)
            grad_list, tg_list , h_list = [], [], []
    
            for ii in range(self.num_classes):
                idx = targets == ii 
                fg_grad, fg_tg, fg_h = grad_p[idx], targets[idx], h[idx]
                ridx = torch.randperm(fg_tg.shape[0])
                s1, s2, s3 = fg_tg[ridx][:mcnt], fg_h[ridx][:mcnt], fg_grad[ridx][:mcnt]
                tg_list.append(s1)
                h_list.append(s2)
                grad_list.append(s3)
                
            grad_p, targets, h = torch.cat(grad_list), torch.cat(tg_list), torch.cat(h_list)     
        
        # print (targets.shape, fg_idx.shape, grad_p.shape, h.shape)
        for ii in range(self.num_classes):
            self.class_weights[ii] += torch.sum(targets == ii)
        
        self.grad_p_sum += grad_p.sum(dim=0)
        self.sample_num += grad_p.shape[0]
        self.curr_constraints += h.sum(dim=0)

        # print ("Gradient shape", grad_p.shape, target_weights.shape, grad_pw.shape, self.grad_p_sum.shape)

    def set_lambd(self, epoch):
        if (epoch + 1) % self.lambd_step == 0:
            
            print (self.class_weights, self.sample_num)
            
            self.class_weights = self.sample_num / self.class_weights
            self.class_weights = self.class_weights / self.class_weights.sum()
            
            grad_p_sumw = self.grad_p_sum * self.class_weights
            
            grad_p_mean = grad_p_sumw / self.sample_num
            
            if dist.is_initialized():
                grad_p_mean = reduce_tensor(grad_p_mean, dist.get_world_size())
                
            self.lambd = torch.clamp(grad_p_mean, min=self.lambd_min, max=self.lambd_max).detach()
            print ("Lambda", self.lambd)

    def update_rho(self, epoch):
        if self.rho_update:
            
            ## this might not be needed.
            # self.curr_constraints = self.curr_constraints * self.class_weights 
            self.curr_constraints = self.curr_constraints / self.sample_num
            
            if dist.is_initialized():
                self.curr_constraints = reduce_tensor(self.curr_constraints, dist.get_world_size())

            if (epoch + 1) % self.rho_step == 0 and self.prev_constraints is not None:
                # increase rho if the constraint became unsatisfied or didn't decrease as expected
                self.rho = torch.where(
                    self.curr_constraints > (self.prev_constraints.clamp(min=0) * self.tao),
                    self.gamma * self.rho,
                    self.rho
                )
                self.rho = torch.clamp(self.rho, min=self.rho_min, max=self.rho_max).detach()

            self.prev_constraints = self.curr_constraints
            print ("Rho", self.rho)

    def get_lambd_metric(self):
        lambd = self.lambd

        return lambd, lambd.mean().item(), lambd.max().item(), lambd.min().item()

    def get_rho_metric(self):
        rho  = self.rho
        return rho, rho.mean().item(), rho.max().item(), rho.min().item()
