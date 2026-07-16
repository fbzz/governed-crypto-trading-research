from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lookback: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, output_dim)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.position.shape[1]:
            raise ValueError("Input sequence exceeds configured lookback")
        hidden = self.input_projection(x) + self.position[:, : x.shape[1]]
        hidden = self.encoder(hidden)
        return self.output(self.norm(hidden[:, -1]))


@dataclass
class TrainingResult:
    model: TemporalTransformer
    best_epoch: int
    train_loss: float
    validation_loss: float


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    model_config: dict,
    seed: int,
) -> TrainingResult:
    seed_everything(seed)
    device = torch.device("cpu")
    model = TemporalTransformer(
        input_dim=x_train.shape[-1],
        output_dim=y_train.shape[-1],
        lookback=x_train.shape[1],
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        n_layers=int(model_config["n_layers"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    objective = model_config.get("objective", "huber_regression")
    if objective == "huber_regression":
        criterion: nn.Module = nn.HuberLoss(delta=0.01)
        train_target = torch.from_numpy(y_train)
        validation_target = torch.from_numpy(y_validation).to(device)
    elif objective == "cross_entropy_top_asset":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(model_config.get("label_smoothing", 0.0))
        )
        train_target = torch.from_numpy(y_train.argmax(axis=1)).long()
        validation_target = torch.from_numpy(y_validation.argmax(axis=1)).long().to(device)
    else:
        raise ValueError(f"Unsupported Transformer objective: {objective}")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), train_target),
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(x_validation).to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    last_train_loss = float("nan")
    for epoch in range(int(model_config["epochs"])):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.to(device))
            loss = criterion(prediction, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        last_train_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_target))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(model_config["early_stopping"]):
                break
    if best_state is None:
        raise RuntimeError("Transformer training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(model, best_epoch, last_train_loss, best_loss)


def predict_transformer(model: TemporalTransformer, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).cpu().numpy()


def save_checkpoint(
    result: TrainingResult,
    path: str | Path,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "best_epoch": result.best_epoch,
            "validation_loss": result.validation_loss,
            "metadata": metadata,
        },
        path,
    )
