import math
import time
import datetime
import torch
import logging
import os
import webdataset as wds
from common.logger import MetricLogger, SmoothedValue
from common.dataset import prepare_sample, reorg_datasets_by_split, concat_datasets
from torch.utils.data.dataset import ChainDataset

class ConstantLRScheduler:
    def __init__(self, optimizer, init_lr, warmup_start_lr=-1, warmup_steps=0, **kwargs):
        self.optimizer = optimizer
        self.lr = init_lr
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.warmup_steps = warmup_steps
    
    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.lr,
            )
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class RunnerIter():
    """
    Run training based on the number of iterations. This is common when
    the training dataset size is large. Underhood logic is similar to
    epoch-based training by considering every #iters_per_inner_epoch as an
    inner epoch.

    In iter-based runner, after every #iters_per_inner_epoch steps, we

        1) do a validation epoch;
        2) schedule the learning rate;
        3) save the checkpoint.

    We refer every #iters_per_inner_epoch steps as an inner epoch.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets
        self.model = model

        self.start_iters = 0
        self.setup_output_dir()

        self.max_iters = int(self.config['run']['max_iters'])
        assert self.max_iters > 0, "max_iters must be greater than 0."

        self.iters_per_inner_epoch = int(
            self.config['run']['iters_per_inner_epoch']
        )
        assert (
            self.iters_per_inner_epoch > 0
        ), "iters_per_inner_epoch must be greater than 0."
        self.lr_scheduler = self.set_lr_scheduler()

        self._optimizer = None
        self._scaler = None



    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            # lr_scale = self.config.run_cfg.get("lr_layer_decay", 1)
            lr_scale = 1
            weight_decay = self.config['run_cfg']["weight_decay"]
            optim_params = self._model.get_optimizer_params(weight_decay,lr_scale)

            num_parameters = 0
            for p_group in optim_params:
                for p in p_group["params"]:
                    num_parameters += p.data.nelement()    
            logging.info("number of trainable parameters: {}".format(num_parameters))      
                  
            # beta2 = self.config.run_cfg.get("beta2", 0.999)
            beta2 = 0.999

            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config['run']['init_lr']),
                betas=(0.9, beta2),
            )    
        return self._optimizer

    @property
    def scaler(self):
        amp = self.config['run']['amp']

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def max_epoch(self):
        return int(self.max_iters / self.iters_per_inner_epoch)

    @property
    def cur_epoch(self):
        try:
            return self.train_loader.epoch
        except AttributeError:
            # pipeline data (e.g. LAION) is streaming, have no concept of epoch
            return 0

    def set_lr_scheduler(self):
        lr_scheduler = ConstantLRScheduler(optimizer=self.optimizer,
                                           init_lr=self.config['run']['init_lr']) # TODO
        return lr_scheduler

    def _progress(self, cur_iters):
        return "{}_iters={}".format(self.cur_epoch, cur_iters)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_iters = 0

        # runner_base.log_config()
        self.log_config()

        # # resume from checkpoint if specified, not used
        # if not self.evaluate_only and self.resume_ckpt_path is not None:
        #     self._load_checkpoint(self.resume_ckpt_path)

        for start_iters in range(
            self.start_iters, self.max_iters, self.iters_per_inner_epoch
        ):
            end_iters = start_iters + self.iters_per_inner_epoch

            logging.info(
                "Start training, max_iters={}, in total {} inner epochs.".format(
                    self.max_iters, int(self.max_iters / self.iters_per_inner_epoch)
                )
            )
            if start_iters == self.start_iters:
                # self.model.before_training
                # self.task.before_training(
                #     model=self.unwrap_dist_model(self.model),
                #     dataset=self.datasets,
                # )
                self.model.before_training(
                    dataset=self.datasets
                )
            train_stats = self.train_iters(self.cur_epoch, start_iters)
            # runner_base.log_stats
            self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase, not used

            self._save_checkpoint(end_iters, is_best=False)

            # dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def train_iters(self, epoch, start_iters):
        # train by iterations
        self.model.train()

        return self.train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_inner_epoch=self.iters_per_inner_epoch,
            model=self.model,
            data_loader=self.train_loader, # TODO
            optimizer=self.optimizer, 
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
        )
    
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        # common.logger
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            # prepare_sample data_utils.py
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
    
    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def _save_checkpoint(self, cur_iters, is_best=False):
        model_no_ddp = self.model
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }

        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "iters": cur_iters,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_iters),
        )
        logging.info("Saving checkpoint at iters {} to {}.".format(cur_iters, save_to))
        torch.save(save_obj, save_to)

    @property
    def dataloaders(self) -> dict:
        if self._dataloaders is None:
            dataset_ratios = None

            
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    try:
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    except TypeError:
                        # a single wds.DataPipeline or ChainDataset
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config['run']['num_workers'],
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders
    
    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"
    
    def setup_output_dir(self):
        output_dir = os.path.join(self.config['run']['output_dir'], self.job_id)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir