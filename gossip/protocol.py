import logging
import random


class GossipProtocol:
    def __init__(self, fanout: int, max_hops: int):
        self.fanout = fanout
        self.max_hops = max_hops

        # track forwarded state: (origin_client_id, forwarder_client_id)
        self._seen_forward: set[tuple[str, str]] = set()
        self.gossip_timings: list[dict] = []

    def reset_round(self):
        self._seen_forward.clear()
        self.gossip_timings.clear()
        logging.info("Gossip round state reset")

    def spread(
        self,
        origin_node,
        all_nodes,
        message: dict,
        hop: int = 0,
    ):
        origin_client_id = message["client_id"]
        state_id = (origin_client_id, origin_node.client_id)

        if state_id in self._seen_forward:
            logging.info(
                f"Gossip message from {origin_client_id} already forwarded by "
                f"{origin_node.client_id}, skipping"
            )
            return

        if hop >= self.max_hops:
            logging.info(f"Max hops reached for message from {origin_client_id}")
            return

        self._seen_forward.add(state_id)

        peers = [n for n in all_nodes if n.client_id != origin_node.client_id]
        if not peers:
            return

        targets = random.sample(peers, min(self.fanout, len(peers)))

        for target in targets:
            self.gossip_timings.append({
                "from": origin_node.client_id,
                "to": target.client_id,
                "origin": origin_client_id,
                "hop": hop + 1,
                "accepted": True,
            })

            logging.info(
                f"[gossip] {origin_node.client_id} -> {target.client_id} "
                f"hop={hop + 1} [FORWARDED]"
            )

            target.receive_gossip(message)
            self.spread(target, all_nodes, message, hop=hop + 1)

    def run_round(self, nodes):
        self.reset_round()

        for node in nodes:
            if node.own_submission is None:
                raise RuntimeError(
                    f"{node.client_id} has no submission — call prepare_update() first"
                )

            logging.info(f"[gossip] spreading update from {node.client_id}")
            self.spread(
                origin_node=node,
                all_nodes=nodes,
                message=node.own_submission,
                hop=0,
            )

    def print_gossip_summary(self):
        if not self.gossip_timings:
            logging.info("No gossip records available for this round")
            return

        logging.info("-" * 70)
        logging.info(f"Gossip log (fanout={self.fanout} max_hops={self.max_hops})")
        logging.info("-" * 70)
        logging.info(
            f"{'Origin':<12} {'From':<12} {'To':<12} {'Hop':<5} Accepted"
        )
        logging.info("-" * 70)

        for t in self.gossip_timings:
            logging.info(
                f"{t['origin']:<12} {t['from']:<12} {t['to']:<12} "
                f"{t['hop']:<5} {t['accepted']}"
            )

        logging.info(f"Total gossip hops: {len(self.gossip_timings)}")