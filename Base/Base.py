import copy
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from dgl.dataloading import GraphDataLoader
from torchmetrics import Metric
from dgl.data import Subset
from torch import nn
from typing import Type
from typing import Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod
# from lab.checker import expected_mean_readout, expected_gin_layer_output, expected_sage_layer_output, \
#     expected_attention_readout, expected_gine_layer_output, expected_sum_readout, expected_simple_mpnn_output


class LoggerBase(ABC):
    def __init__(self, logdir: str | Path):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        with open(self.logdir / "logs.txt", 'wb') as f:
            f.write(metrics)

    @abstractmethod
    def close(self):
        self.logdir.close()


class DummyLogger(LoggerBase):  # If you don't want to use any logger, you can use this one
    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        pass

    def close(self):
        pass

    def restart(self):
        pass


class MetricList:
    def __init__(self, metrics: Dict[str, Metric]):
        self.metrics = copy.deepcopy(metrics)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for name, metric in self.metrics.items():
            metric.update(preds.detach().cpu(), targets.cpu())

    def compute(self) -> Dict[str, float]:
        metrics = {}
        for name, metric_fn in self.metrics.items():
            metrics[name] = metric_fn.compute().item()
            metric_fn.reset()
        return metrics


class TrainerBase:
    def __init__(
            self,
            *,
            run_dir: str | Path,
            train_dataset: Subset,
            valid_dataset: Subset,
            train_metrics: Dict[str, Metric],
            valid_metrics: Dict[str, Metric],
            model: nn.Module,
            logger: LoggerBase,
            optimizer_kwargs: Dict[str, Any],
            optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
            n_epochs: int,
            train_batch_size: int = 32,
            valid_batch_size: int = 16,
            device: str = "cuda",
            valid_every_n_epochs: int = 1,
            loss_fn=nn.MSELoss()
    ):
        self.run_dir = Path(run_dir)
        self.train_loader = GraphDataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
        )
        self.valid_loader = GraphDataLoader(
            dataset=valid_dataset,
            batch_size=valid_batch_size,
            shuffle=True,
        )
        self.train_metrics = MetricList(train_metrics)
        self.valid_metrics = MetricList(valid_metrics)
        self.logger = logger
        self.model = model
        self.optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        self.n_epochs = n_epochs
        self.device = device
        self.valid_every_n_epochs = valid_every_n_epochs
        self.loss_fn = loss_fn
        self.model.to(device)

    @torch.no_grad()
    def validate(self, dataloader: GraphDataLoader, prefix: str) -> Dict[str, float]:
        previous_mode = self.model.training
        self.model.eval()
        losses = []
        for _, graphs, labels in dataloader:
            graphs = graphs.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(graphs)
            loss = self.loss_fn(preds, labels)
            losses.append(loss.item())
            self.valid_metrics.update(preds, labels)
        self.model.train(mode=previous_mode)
        metrics = {"loss": np.mean(losses)} | self.valid_metrics.compute()
        self.logger.log_metrics(metrics=metrics, prefix=prefix)
        return metrics

    def train(self) -> Dict[str, float]:
        self.model.train()
        valid_metrics = {}
        for epoch in tqdm(range(self.n_epochs), total=self.n_epochs):
            for _, graphs, labels in self.train_loader:
                self.optimizer.zero_grad()
                graphs = graphs.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(graphs)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()

                self.train_metrics.update(preds, labels)
                train_metrics = {"loss": loss.item()} | self.train_metrics.compute()
                self.logger.log_metrics(metrics=train_metrics, prefix="train")

                if epoch % self.valid_every_n_epochs == 0 or epoch == self.n_epochs - 1:
                    valid_metrics = self.validate(self.valid_loader, prefix="valid")

        return valid_metrics

    def test(self, dataset: Subset) -> Dict[str, float]:
        dataloader = GraphDataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=False,
        )
        return self.validate(dataloader, prefix="test")

    def close(self):  # close the logger, not really required for wandb
        self.logger.close()


class ReadoutBase(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    @abstractmethod
    def forward(self,
                node_embeddings: torch.Tensor,
                graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Attributes:
            node_embeddings: node embeddings in a sparse format, i.e. [total_num_nodes, hidden_size]
            graph: a DGLGraph that contains the graph structure
        Returns:
            graph_embeddings: graph embeddings of shape.[batch_size, hidden_size]
        """
        raise NotImplementedError()