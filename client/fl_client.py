import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

from model.cnn import SmallCNN
from utils.weights import apply_weight_arrays, weights_to_bytes


def build_model(
    model_name: str,
    device: str,
    input_channels: int,
    num_classes: int,
    input_height: int,
    input_width: int,
    conv1_channels: int,
    conv2_channels: int,
    hidden_dim: int,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "smallcnn":
        return SmallCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        ).to(device)

    raise ValueError(f"Unsupported model: {model_name}")


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        device: str,
        weight_dtype: str,
        learning_rate: float,
        model_name: str,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        hidden_dim: int,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.weight_dtype = weight_dtype
        self.learning_rate = learning_rate
        self.model_name = model_name

        # Differential Privacy parameters
        self.dp_clip_norm = 1.0
        self.dp_noise_std = 0.01

        self.model = build_model(
            model_name=self.model_name,
            device=self.device,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self.dp_noise_std,
                max_grad_norm=self.dp_clip_norm,
            )
            logging.info(
                f"[{client_id}] REAL DP enabled via Opacus | "
                f"clip_norm={self.dp_clip_norm} noise_multiplier={self.dp_noise_std}"
            )
        else:
            logging.warning(
                f"[{client_id}] Opacus not installed! Falling back to FAKE batch-level DP. "
                "Please run `pip install opacus` for real Differential Privacy."
            )

        logging.info(
            f"[{client_id}] initialized | "
            f"model={self.model_name} learning_rate={self.learning_rate} "
            f"weight_dtype={self.weight_dtype}"
        )

    def local_train(self, global_weight_arrays=None, epochs=1):
        if global_weight_arrays is not None:
            apply_weight_arrays(self.model, global_weight_arrays)

        if epochs == 0:
            return

        self.model.train()
        total_loss = 0.0

        for _ in range(epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss.backward()

                if not OPACUS_AVAILABLE:
                    # Fallback: FAKE batch-level DP
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.dp_clip_norm,
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(
                                mean=0.0,
                                std=self.dp_noise_std,
                                size=param.grad.shape,
                                device=param.grad.device,
                            )
                            param.grad += noise

                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx == 0:
                    pred = torch.argmax(logits, dim=1)
                    logging.info(
                        f"[{self.client_id}] pred={pred[0].item()} | actual={y[0].item()}"
                    )

        total_batches = len(self.dataloader) * epochs
        logging.info(
            f"[{self.client_id}] trained with DP | "
            f"loss: {total_loss / total_batches:.4f}"
        )

    def prepare_update(self) -> dict:
        """
        Prepare plain model update.
        No Dilithium.
        No ZKP.
        DP is applied during local training.
        """
        update_bytes = weights_to_bytes(self.model, self.weight_dtype)

        logging.info(
            f"[{self.client_id}] update prepared | size={len(update_bytes)/1024:.1f} KB"
        )

        return {
            "client_id": self.client_id,
            "update_bytes": update_bytes,
        }