import logging
import yaml
import time
import random
import torch

from data.loader import make_client_loaders
from gossip.node import GossipNode
from gossip.protocol import GossipProtocol
from utils.weights import model_to_weight_arrays


def setup_logging(config):
    logging.basicConfig(
        level=getattr(logging, config["logging"]["log_level"].upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["log_file"]),
            logging.StreamHandler()
        ]
    )


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def choose_aggregator_node(nodes):
    counts = []

    for node in nodes:
        subs = node.get_all_submissions()
        count = len(subs)
        counts.append((node, count))
        logging.info(f"{node.client_id} submissions = {count}")

    max_count = max(c for _, c in counts)
    candidates = [n for n, c in counts if c == max_count]

    aggregator = random.choice(candidates)
    logging.info(f"Selected aggregator: {aggregator.client_id}")
    return aggregator


def sync_weights_to_all_nodes(nodes, weights):
    for node in nodes:
        node.local_train(weights, epochs=0)


def clear_round_state(nodes, gossip):
    for node in nodes:
        node.clear_submissions()
    gossip.reset_round()


def main():
    config = load_config()
    setup_logging(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # -------- CONFIG --------
    N_CLIENTS = config["experiment"]["n_clients"]
    N_ROUNDS = config["experiment"]["n_rounds"]
    LOCAL_EPOCHS = config["experiment"]["local_epochs"]

    GOSSIP_FANOUT = config["gossip"]["fanout"]
    GOSSIP_MAX_HOPS = config["gossip"]["max_hops"]

    LEARNING_RATE = config["training"]["learning_rate"]

    MODEL = config["model"]
    DATA = config["data"]
    WEIGHTS = config["weights"]

    # -------- DATA --------
    client_loaders, _ = make_client_loaders(
        n_clients=N_CLIENTS,
        batch_size=DATA["batch_size"],
        alpha=DATA["alpha"],
        dataset_name=DATA["dataset_name"],
        partition_by=DATA["partition_by"],
        min_partition_size=DATA["min_partition_size"],
        self_balancing=DATA["self_balancing"],
        seed=DATA["seed"],
        test_batch_size=DATA["test_batch_size"],
        normalize_mean=DATA["normalize_mean"],
        normalize_std=DATA["normalize_std"],
    )

    # -------- NODES --------
    nodes = []
    for i in range(N_CLIENTS):
        node = GossipNode(
            client_id=f"client_{i}",
            dataloader=client_loaders[i],
            device=device,
            learning_rate=LEARNING_RATE,
            model_name=MODEL["name"],
            weight_dtype=WEIGHTS["dtype"],
            input_channels=MODEL["input_channels"],
            num_classes=MODEL["num_classes"],
            input_height=MODEL["input_height"],
            input_width=MODEL["input_width"],
            conv1_channels=MODEL["conv1_channels"],
            conv2_channels=MODEL["conv2_channels"],
            hidden_dim=MODEL["hidden_dim"],
        )
        nodes.append(node)

    # -------- GOSSIP --------
    # No Dilithium, no public keys, no ZKP.
    gossip = GossipProtocol(
        fanout=GOSSIP_FANOUT,
        max_hops=GOSSIP_MAX_HOPS,
    )

    # -------- INITIAL MODEL SYNC --------
    initializer = random.choice(nodes)
    init_weights = model_to_weight_arrays(initializer.client.model)
    sync_weights_to_all_nodes(nodes, init_weights)
    logging.info(f"Initial model taken from {initializer.client_id} and synced to all nodes")

    # -------- TRAINING --------
    start_time = time.time()

    for r in range(1, N_ROUNDS + 1):
        logging.info("=" * 60)
        logging.info(f"Round {r}/{N_ROUNDS}")
        logging.info("=" * 60)

        clear_round_state(nodes, gossip)

        # 1. Local training
        for node in nodes:
            node.local_train(None, epochs=LOCAL_EPOCHS)

        # 2. Prepare plain updates without Dilithium/ZKP
        for node in nodes:
            node.prepare_update()

        # 3. Gossip propagation
        gossip.run_round(nodes)

        # 4. Choose aggregator
        aggregator = choose_aggregator_node(nodes)
        subs = aggregator.get_all_submissions()

        # 5. Aggregate
        if len(subs) == N_CLIENTS:
            logging.info(
                f"[{aggregator.client_id}] aggregating complete round "
                f"({len(subs)}/{N_CLIENTS} submissions)"
            )
        else:
            logging.warning(
                f"[{aggregator.client_id}] incomplete aggregation: "
                f"{len(subs)}/{N_CLIENTS} submissions available"
            )

        if len(subs) > 0:
            aggregator.aggregate_local_updates(subs, aggregator.client.model)

            weights = model_to_weight_arrays(aggregator.client.model)
            sync_weights_to_all_nodes(nodes, weights)

            logging.info(f"Round {r} aggregated model synced to all nodes")
        else:
            logging.warning(f"Round {r} skipped because no submissions were available")

        clear_round_state(nodes, gossip)

    end_time = time.time()
    logging.info(f"Total time = {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()