import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from torch import distributed as dist
import torch
import torch.nn.functional as F

from .penalty import get_penalty_func
from .aug_lagrangian import AugLagrangian
# from ..utils import reduce_tensor

logger = logging.getLogger(__name__)


class SpatialBndAugLagrangian(AugLagrangian):
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
        normalize: bool = False,
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
        self.is_undersample = is_undersample

        self.ncases = 2 ## uniform, boundary.
        self.lambd = self.lambd_min * torch.ones(self.ncases, requires_grad=False).cuda()
        # self.rho = self.rho_min
        # class-wise rho
        self.rho = self.rho_min * torch.ones(self.ncases, requires_grad=False).cuda()
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
        # constraints = torch.abs(self.margin * rmask - logits).mean(1)
        constraints_easy = torch.abs(self.margin * rmask - logits).mean(1)
        constraints_diff = torch.abs(rmask - logits).mean(1)
        # print (constraints_easy.shape, constraints_diff.shape)
        # max_values = logits.amax(dim=dim, keepdim=True)
        # diff = max_values - logits
        # constraints = diff - self.margin
        # if self.normalize:
        #     constraints = constraints / max_values
        
        umask_mean = torch.mean(utarget,dim=1,keepdim=True)
        diff_sum = (umask_mean - utarget).sum(dim=1,keepdim=True)
        reg_idx = diff_sum != 0
        reg_idx = reg_idx.reshape(bs, h, w).int()
        constraints = reg_idx * constraints_diff + (1 - reg_idx) * constraints_easy
        reg_idx = (targets > 0).int() ## foreground. 
        constraints = constraints.unsqueeze(-1) 
        reg_idx = reg_idx.unsqueeze(-1)

        return constraints, reg_idx 
    
    def get(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h,reg_idx = self.get_constraints(logits, targets)
        # logits = logits.movedim(1, -1)  # move class dimension to last
        # h = h.movedim(1, -1) ## to have same dimension as that of logit.
        # print (h.shape, reg_idx.shape)
        hcat = torch.cat([h * reg_idx, h * (1-reg_idx)], -1)
        p, _ = self.penalty_func(hcat, self.lambd, self.rho)
        # penalty = p.sum(dim=-1).mean()  # sum over classes and average over samples (and possibly pixels)
        penalty = p.mean()
        constraint = h.mean()
        
        return penalty, constraint
    
    def reset_update_lambd(self, epoch):
        # if (epoch + 1) % self.lambd_step == 0:
        self.grad_p_sum = torch.zeros_like(self.lambd)
        self.sample_num = 0
        self.curr_constraints = torch.zeros_like(self.rho)
        # self.case_weights = torch.zeros_like(self.lambd)

    def update_lambd(self, logits, targets, epoch):
        """update lamdb based on the gradeint on the logits
        """
        h, reg_idx = self.get_constraints(logits, targets)
        # if (epoch + 1) % self.lambd_step == 0:
        # logits = logits.movedim(1, -1)  # move class dimension to last
        # h = h.movedim(1, -1) ## to have same dimension as that of logit.
        
        hcat = torch.cat([h * reg_idx, h * (1-reg_idx)],-1)
        
        # for ii in range(self.ncases):
        #     self.case_weights[ii] += torch.sum(reg_idx == ii)

        _, grad_p = self.penalty_func(h, self.lambd, self.rho)
        grad_p = torch.clamp(grad_p, min=self.lambd_min, max=self.lambd_max)
        grad_p = grad_p.flatten(start_dim=0, end_dim=-2)
        self.grad_p_sum += grad_p.sum(dim=0)
        self.sample_num += grad_p.shape[0]
        hcat = hcat.flatten(start_dim=0, end_dim=-2)
        self.curr_constraints += hcat.sum(dim=0)

    def set_lambd(self, epoch):
        if (epoch + 1) % self.lambd_step == 0:
            
            # self.case_weights = self.sample_num / self.case_weights
            # self.case_weights = self.case_weights / self.case_weights.sum()           
            # grad_p_sumw = self.grad_p_sum * self.case_weights

            grad_p_mean = self.grad_p_sum / self.sample_num
            
            if dist.is_initialized():
                grad_p_mean = reduce_tensor(grad_p_mean, dist.get_world_size())
            self.lambd = torch.clamp(grad_p_mean, min=self.lambd_min, max=self.lambd_max).detach()

    def update_rho(self, epoch):
        if self.rho_update:
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

    def get_lambd_metric(self):
        lambd = self.lambd

        return lambd, lambd.mean().item(), lambd.max().item(), lambd.min().item()

    def get_rho_metric(self):
        rho = self.rho
        return rho, rho.mean().item(), rho.max().item(), rho.min().item()
