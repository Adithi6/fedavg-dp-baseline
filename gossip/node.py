import logging
from torch.utils.data import DataLoader
import numpy as np

from client.fl_client import FederatedClient
from utils.weights import bytes_to_weight_arrays, apply_weight_arrays


class GossipNode:
    """
    A GossipNode = FederatedClient + gossip inbox.
    Baseline version:
    - No Dilithium
    - No ZKP
    - Keeps gossip communication
    - Keeps decentralized aggregation
    """

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
        self.client = FederatedClient(
            client_id=client_id,
            dataloader=dataloader,
            device=device,
            weight_dtype=weight_dtype,
            learning_rate=learning_rate,
            model_name=model_name,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        )

        self.client_id = client_id

        self.own_submission: dict | None = None
        self.inbox: dict[str, dict] = {}

        logging.info(
            f"[{self.client_id}] gossip node initialized | "
            f"weight_dtype={weight_dtype} learning_rate={learning_rate} "
            f"model={model_name}"
        )

    def local_train(self, global_weight_arrays: list | None, epochs: int = 1):
        self.client.local_train(global_weight_arrays, epochs)

    def prepare_update(self) -> dict:
        """
        Creates a normal model update without Dilithium signature or ZKP proof.
        """
        self.own_submission = self.client.prepare_update()
        self.inbox.clear()

        logging.info(f"[{self.client_id}] plain update prepared and inbox reset")
        return self.own_submission

    def receive_gossip(self, message: dict):
        origin_id = message["client_id"]

        if origin_id == self.client_id:
            logging.warning(
                f"[{self.client_id}] ignored returned own gossip from {origin_id}"
            )
            return

        if origin_id in self.inbox:
            logging.warning(
                f"[{self.client_id}] duplicate gossip ignored from {origin_id}"
            )
            return

        self.inbox[origin_id] = message

        logging.info(
            f"[{self.client_id}] received gossip from {origin_id} "
            f"| inbox_size={len(self.inbox)}"
        )

    def get_all_submissions(self) -> list[dict]:
        all_subs = []

        if self.own_submission is not None:
            all_subs.append(self.own_submission)

        all_subs.extend(self.inbox.values())
        return all_subs

    def clear_submissions(self):
        self.own_submission = None
        self.inbox.clear()
        logging.info(f"[{self.client_id}] cleared round submissions")

    def aggregate_local_updates(self, submissions: list[dict], template_model):
        if not submissions:
            logging.warning(f"[{self.client_id}] no submissions available for aggregation")
            return

        logging.info(f"[{self.client_id}] aggregating {len(submissions)} submission(s)")

        dtype_name = self.client.weight_dtype

        weight_sets = []
        for sub in submissions:
            arrays = bytes_to_weight_arrays(
                sub["update_bytes"],
                template_model,
                dtype_name=dtype_name,
            )
            weight_sets.append(arrays)

        averaged = [
            np.mean([weights[i] for weights in weight_sets], axis=0)
            for i in range(len(weight_sets[0]))
        ]

        apply_weight_arrays(self.client.model, averaged)

        logging.info(f"[{self.client_id}] local aggregation completed")