from typing import Dict
import numpy as np
import os.path as osp
from shutil import copyfile
import time
import json
import logging
import torch
import torch.nn.functional as F
# import torchvision
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
from terminaltables.ascii_table import AsciiTable

from calibrate.engine.trainer import Trainer
from calibrate.net import ModelWithTemperature
from calibrate.losses import LogitMarginL1
from calibrate.evaluation import (
    AverageMeter, LossMeter, SegmentEvaluator,
    SegmentCalibrateEvaluator,
    CalibSegmentEvaluator
)
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)
from calibrate.utils.torch_helper import to_numpy, get_lr, grid_image
from tqdm import tqdm
from calibrate.utils.bw_const_helper import transform_batch, get_lambda_maps

logger = logging.getLogger(__name__)


class LagrangianMedSegmentTrainer(Trainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.init_lagrangian()

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        if hasattr(self.loss_func, "names"):
            self.loss_meter = LossMeter(
                num_terms=len(self.loss_func.names),
                names=self.loss_func.names
            )
        else:
            self.loss_meter = LossMeter()

        self.calibrate_evaluator = SegmentCalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            ignore_index=255,
            device=self.device
        )
        
        self.testevaluator = CalibSegmentEvaluator(
            self.test_loader.dataset.classes,
            ignore_index=255,
            ishd=True,
            dataset_type=self.cfg.data.name
        )
        # self.logits_evaluator = SegmentLogitsEvaluator(ignore_index=255)
        self.penalty_meter = AverageMeter()
        self.constraint_meter = AverageMeter()


    def reset_meter(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()
        self.testevaluator.reset()
        # self.logits_evaluator.reset()
        self.penalty_meter.reset()
        self.constraint_meter.reset()
        
    def init_lagrangian(self) -> None:
        self.lagrangian = instantiate(self.cfg.lag.object)

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())

        log_dict["penalty"] = self.penalty_meter.val
        log_dict["constraint"] = self.constraint_meter.val        

        # log_dict.update(self.logits_evaluator.curr_score())
        # log_dict.update(self.probs_evaluator.curr_score())
        logger.info("{} Iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable and phase.lower() == "train":
            wandb_log_dict = {"iter": epoch * max_iter + iter}
            wandb_log_dict.update(dict(
                ("{}/Iter/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_epoch_info(self, epoch, phase="Train"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, LogitMarginL1):
            log_dict["alpha"] = self.loss_func.alpha
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
    
        log_dict["penalty"] = self.penalty_meter.avg
        log_dict["constraint"] = self.constraint_meter.avg
        lambd, log_dict["lambd_mean"], log_dict["lambd_max"], log_dict["lambd_min"] = self.lagrangian.get_lambd_metric()
        rho, log_dict["rho_mean"], log_dict["rho_max"], log_dict["rho_min"] = self.lagrangian.get_rho_metric()
        logger.info("Lambd: {}, Rho: {}".format(lambd, rho))

        # log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_eval_epoch_info(self, epoch, phase="Val"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict.update(self.loss_meter.get_avgs())
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        if phase.lower() == "test":
            calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(isprint=False)
            log_dict.update(calibrate_metric)
        # log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        class_table_data = self.evaluator.class_score(isprint=True, return_dataframe=True)
        if phase.lower() == "test":
            logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["{}/segment_score_table".format(phase)] = (
                wandb.Table(
                    dataframe=class_table_data
                )
            )
            if phase.lower() == "test":
                wandb_log_dict["{}/calibrate_score_table".format(phase)] = (
                    wandb.Table(
                        columns=calibrate_table_data[0],
                        data=calibrate_table_data[1:]
                    )
                )
            if "test" in phase.lower() and self.cfg.calibrate.visualize:
                fig_reliab, fig_hist = self.calibrate_evaluator.plot_reliability_diagram()
                wandb_log_dict["{}/calibrate_reliability".format(phase)] = fig_reliab
                wandb_log_dict["{}/confidence_histogram".format(phase)] = fig_hist
                
            wandb.log(wandb_log_dict)    

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # print (inputs.shape, torch.unique(labels))

            # forward
            outputs = self.model(inputs)
            
            
            # print (outputs.shape, labels.shape)

            if isinstance(outputs, Dict):
                outputs = outputs["out"]
                
            if self.cfg.loss.name == 'adaptive_margin_svls':
                loss = self.loss_func(outputs, labels, inputs)
            else:
                loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                # For compounding loss, make sure the first term is the overall loss
                loss_total = loss[0]
            else:
                loss_total = loss
            
            if self.cfg.lag.name == 'spatial_bndry_cls_aug_lag_dst':
                dst_map = torch.from_numpy(get_lambda_maps(labels.cpu()[:,None], self.num_classes)).to(self.device)    
                penalty, constraint = self.lagrangian.get(outputs, labels, dst_map)
            else:
                penalty, constraint = self.lagrangian.get(outputs, labels)
            # backward
            
            self.optimizer.zero_grad()
            (loss_total + penalty).backward()
            # loss_total.backward()

            if self.cfg.train.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = F.softmax(outputs, dim=1)
            
            # if self.cfg.model.name == 'nnunet':
            #     predicts = predicts[:,0]
                
            pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                pred_labels.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            self.penalty_meter.update(penalty.item())
            self.constraint_meter.update(constraint.item())
            # self.logits_evaluator.update(to_numpy(outputs), to_numpy(labels))
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()

            # break

        self.log_epoch_info(epoch)

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="Val"):
        self.reset_meter()
        self.model.eval()

        if phase.lower() == "val":
            self.lagrangian.reset_update_lambd(epoch)


        max_iter = len(data_loader)
        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            inputs = inputs.squeeze()
            labels = labels.squeeze()
            
            inputs = inputs.unsqueeze(1)
            
            # print (inputs.shape, labels.shape)
            # dst_map = torch.from_numpy(get_lambda_maps(labels.cpu()[:,None], self.num_classes)).to(self.device)
            
            outputs = self.model(inputs)
            
            # print (inputs.shape, outputs.shape)
            
            # print (outputs.shape)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
                
            if self.cfg.loss.name == 'adaptive_margin_svls':
                loss = self.loss_func(outputs, labels, inputs)
            else:
                loss = self.loss_func(outputs, labels)

            if phase.lower() == "val":
                
                # self.lagrangian.update_lambd(outputs, labels, dst_map, epoch)
                # penalty, constraint = self.lagrangian.get(outputs, labels, dst_map)
                
                self.lagrangian.update_lambd(outputs, labels, epoch)
                penalty, constraint = self.lagrangian.get(outputs, labels)
                
                self.penalty_meter.update(penalty.item())
                self.constraint_meter.update(constraint.item())

            # metric
            self.loss_meter.update(loss)
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                to_numpy(pred_labels),
                to_numpy(labels)
            )
            # if phase.lower() == "test":
            #     self.calibrate_evaluator.update(
            #         outputs, labels
            #     )
            # self.logits_evaluator(
            #     np.expand_dims(to_numpy(outputs), axis=0),
            #     np.expand_dims(to_numpy(labels), axis=0)
            # )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            # if (i + 1) % self.cfg.log_period == 0:
            #     self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()

            # break
        if phase.lower() == "val":
            self.lagrangian.set_lambd(epoch)
            self.lagrangian.update_rho(epoch)
            
        self.log_eval_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(main=True)


    
    @torch.no_grad()
    def disp_epoch(self, data_loader, epoch, phase="Val"):

        self.model.eval()

        for i, (inputs, labels) in enumerate(data_loader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            pred_labels = torch.argmax(F.softmax(outputs,dim=1), dim=1)

            input_grid = grid_image(inputs)
            label_grid = grid_image(labels.unsqueeze(1))
            pred_grid = grid_image(pred_labels.unsqueeze(1))
            # pred_grid = grid_image(outputs)

            if self.cfg.wandb.enable:
                wandb.log({"Input":wandb.Image(input_grid),"Target":wandb.Image(label_grid),"Prediction":wandb.Image(pred_grid)})
        
            # break

        return

    @torch.no_grad()
    def test_epoch(self, data_loader,phase='Test'):
        self.reset_meter()
        self.model.eval()

        end = time.time()
        
        for i, (inputs, labels, fpath) in enumerate(tqdm(data_loader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.squeeze()
            labels = labels.squeeze()
            
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(0).unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
                
            # metric
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
                        
            self.testevaluator.update(
                to_numpy(pred_labels),
                to_numpy(labels),
                outputs,
                labels,
                fpath
            )
            
            self.batch_time_meter.update(time.time() - end)

            end = time.time()
            
        self.log_test_epoch_info(phase)

    def log_test_epoch_info(self, phase='Test'):
        log_dict = {}
        metric = self.testevaluator.mean_score()
        log_dict.update(metric)

        logger.info("{} Epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        class_table_data = self.testevaluator.class_score(isprint=True, return_dataframe=True)
        calibrate_table_data = self.testevaluator.calib_score(isprint=True)
        
        class_hd_list = class_table_data['hd'].to_list()[:-1]
        class_dice_list = class_table_data['dsc'].to_list()[:-1]
        class_name_list = class_table_data['Class'].to_list()[:-1]
        
        self.testevaluator.save_csv(osp.dirname(str(self.cfg.test.checkpoint)))
        
        for ii in range(len(class_name_list)):
            key = 'dsc-{}'.format(class_name_list[ii])
            val = class_dice_list[ii]
            log_dict.update({key:val})    
            
            key = 'hd-{}'.format(class_name_list[ii])
            val = class_hd_list[ii]
            log_dict.update({key:val})
        
        if self.cfg.wandb.enable:
            wandb_log_dict = {}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["{}/segment_score_table".format(phase)] = (
                wandb.Table(
                    dataframe=class_table_data
                )
            )
            if phase.lower() == "test":
                wandb_log_dict["{}/calibrate_score_table".format(phase)] = (
                    wandb.Table(
                        dataframe=calibrate_table_data
                    )
                )
            wandb.log(wandb_log_dict)
    

    def test(self):
        logger.info("We are almost done : final testing ...")
        self.test_loader = instantiate(self.cfg.data.object.test)
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "best.pth"), self.model, self.device
        )
        self.test_epoch(self.test_loader)