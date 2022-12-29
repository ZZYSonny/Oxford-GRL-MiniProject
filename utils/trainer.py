import torch
import torch.nn as nn
import torch_geometric.loader as tgloader
import typing
from dataclasses import dataclass
from tqdm import tqdm
from . import neptune

@dataclass
class ExecutionConfig:
    num_epoch: int
    batch_size: int
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_runs: int = 1


@dataclass
class Trainer:
    # Core
    get_model: typing.Callable[[], nn.Module]
    get_optimizer: typing.Callable[[nn.Module], torch.optim.Optimizer]
    get_scheduler: typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.ReduceLROnPlateau]
    get_dataset: typing.Callable[[], typing.Tuple[list, list, list]]
    find_loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    find_metric: typing.Callable[[torch.Tensor, torch.Tensor], typing.Any]
    exec_config: ExecutionConfig
    early_stop: typing.Callable[[float], bool] = lambda x: False

    def train_one_epoch(self, loader):
        self.model.train()
        
        loss_sum = 0
        cnt = 0
        for batch in tqdm(loader, leave=False, desc="Training"):
            batch = batch.to(self.exec_config.device)
            model_out = self.model(batch)

            self.optimizer.zero_grad()
            loss = self.find_loss(model_out, batch.y)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.detach().cpu().item()
            cnt += 1
            if cnt == 25:
                self.tracker[f"run{self.cur_run}/loss"].log(loss_sum / cnt)
                loss_sum = 0
                cnt = 0
            
    def evaluate_model(self, loader, task):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in tqdm(loader, leave=False, desc=f"Eval {task}"):
                batch = batch.to(self.exec_config.device)
                pred = self.model(batch)
                y_true.append(batch.y.detach().cpu())
                y_pred.append(pred.detach().cpu())

            y_true = torch.cat(y_true, dim = 0)
            y_pred = torch.cat(y_pred, dim = 0)
            metric = self.find_metric(y_pred, y_true)
        
        return metric


    def run_once(self):
        self.model = self.get_model().to(self.exec_config.device)
        self.optimizer = self.get_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer)
        dataset = self.get_dataset()

        train_loader = tgloader.DataLoader(
            dataset[0],
            batch_size=self.exec_config.batch_size, 
            shuffle=True
        )
        valid_loader = tgloader.DataLoader(
            dataset[1],
            batch_size=self.exec_config.batch_size, 
            shuffle=False
        )
        test_loader  = tgloader.DataLoader(
            dataset[2],
            batch_size=self.exec_config.batch_size, 
            shuffle=False
        )

        metric = {
            "train": 0,
            "valid": 0,
            "test": 0,
        }

        bar = tqdm(range(self.exec_config.num_epoch), leave=False, desc="Iter")
        for epoch in bar:
            self.train_one_epoch(train_loader)
            
            cur_metric = {
                "train": self.evaluate_model(train_loader, "train"),
                "valid": self.evaluate_model(valid_loader, "valid"),
                "test": self.evaluate_model(test_loader, "test")
            }
            self.scheduler.step(cur_metric["valid"])

            for name in ["train", "valid", "test"]:
                metric[name] = max(cur_metric[name], metric[name])
                self.tracker[f"run{self.cur_run}/{name}_history"].log(cur_metric[name])
            
            bar.set_description(str(cur_metric))
            if self.early_stop(cur_metric["valid"]):
                self.evaluate_model(test_loader, "test")
                break
        
        return metric

    def run(self):
        self.tracker = neptune.init_tracker()
        for i in range(self.exec_config.num_runs):
            self.cur_run = i
            self.run_once()