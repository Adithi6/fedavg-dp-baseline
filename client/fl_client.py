import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

        import math

        # Differential Privacy parameters
        self.dp_clip_norm = 1.0
        self.epsilon = 5.0   # Fine-tuned privacy budget (Balanced: strong privacy, decent accuracy)
        self.delta = 1e-5    # Usually set to 1/N (where N is size of dataset)

        # Calculate noise_std mathematically using the Gaussian mechanism formula:
        # sigma = (clip_norm * sqrt(2 * ln(1.25 / delta))) / epsilon
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        self.dp_noise_std = (c * self.dp_clip_norm) / self.epsilon

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

        logging.info(
            f"[{client_id}] DP enabled via mathematical noise addition | "
            f"clip_norm={self.dp_clip_norm} noise_multiplier={self.dp_noise_std}"
        )

        logging.info(
            f"[{client_id}] initialized | "
            f"model={self.model_name} learning_rate={self.learning_rate} "
            f"weight_dtype={self.weight_dtype}"
        )

    def local_train(self, global_weight_arrays=None, epochs=1):
        import time
        start_time = time.time()

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

                # Apply Differential Privacy mathematically
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
        end_time = time.time()
        training_time = end_time - start_time

        logging.info(
            f"[{self.client_id}] trained with DP | "
            f"loss: {total_loss / total_batches:.4f} | "
            f"time: {training_time:.2f}s"
        )

    def prepare_update(self) -> dict:
        """
        Prepare plain model update.
        No Dilithium.
        No ZKP.
        DP is applied during local training.
        """
        import time
        import sys
        start_time = time.time()

        update_bytes = weights_to_bytes(self.model, self.weight_dtype)

        payload = {
            "client_id": self.client_id,
            "update_bytes": update_bytes,
        }
        
        # Approximate size of the whole payload being sent over the network
        payload_size_kb = (sys.getsizeof(self.client_id) + sys.getsizeof(update_bytes)) / 1024.0

        end_time = time.time()
        prep_time = end_time - start_time

        logging.info(
            f"[{self.client_id}] update prepared | payload_size={payload_size_kb:.2f} KB | prep_time={prep_time:.4f}s"
        )

        return payload