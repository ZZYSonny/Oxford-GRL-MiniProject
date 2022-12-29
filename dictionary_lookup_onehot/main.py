import torch
import torch.optim as optim
import sklearn.metrics as skmetrics

from . import data
from models import models
from utils import trainer

loss_func = torch.nn.CrossEntropyLoss()
def find_loss(model_out: torch.Tensor, dataset_y: torch.Tensor):
    return loss_func(model_out, dataset_y)

def find_metric(model_out: torch.Tensor, dataset_y: torch.Tensor):
    y_model = torch.argmax(model_out, dim=1)
    return (y_model==dataset_y).float().mean()

# Configs
n = 20
# HyperParameters
lr = 1e-1
min_lr = 1e-5
weight_decay = 0
lr_reduce_factor = 0.5
lr_reduce_patience = 10

exec_config = trainer.ExecutionConfig(
    num_epoch = 200,
    batch_size = 1024,
)
trainer.Trainer(
    get_model = lambda: models.ModelX(2*n,n),
    get_optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
    get_scheduler=lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=min_lr, factor=lr_reduce_factor, patience=lr_reduce_patience, verbose=True),
    get_dataset = lambda: data.gen_dataset(n, 100000, 1024, 1024),
    find_loss = find_loss,
    find_metric = find_metric,
    exec_config=exec_config,
).run()