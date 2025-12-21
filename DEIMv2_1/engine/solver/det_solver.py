"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch
import wandb

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg
        if dist_utils.is_main_process():

            wandb.init(
                project="DEIMv2",          # 원하는 프로젝트 이름
                name="DEIMv2",             # run 이름
                config=args.yaml_cfg,      # 전체 yaml config 업로드
            )

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if dist_utils.is_main_process():
                # train loss, lr
                train_loss = float(train_stats.get("loss", 0.0))
                lr = float(self.optimizer.param_groups[0]["lr"])

                # coco bbox mAP (AP50:95, AP50) – 있으면 로깅
                ap_list = test_stats.get("coco_eval_bbox", None)
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/lr": lr,
                }
                if ap_list is not None and len(ap_list) >= 12:
                    log_dict["val/AP50_95"]  = float(ap_list[0])
                    log_dict["val/AP50"]     = float(ap_list[1])
                    log_dict["val/AP75"]     = float(ap_list[2])
                    log_dict["val/AP_small"] = float(ap_list[3])
                    log_dict["val/AP_medium"]= float(ap_list[4])
                    log_dict["val/AP_large"] = float(ap_list[5])

                    log_dict["val/AR50_95"]  = float(ap_list[8])
                    log_dict["val/AR_small"] = float(ap_list[9])
                    log_dict["val/AR_medium"]= float(ap_list[10])
                    log_dict["val/AR_large"] = float(ap_list[11])


                wandb.log(log_dict)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if dist_utils.is_main_process():
            wandb.finish()

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
