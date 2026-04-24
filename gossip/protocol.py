import logging
import random
import hashlib

from crypto import dilithium_utils


class GossipProtocol:
    def __init__(
        self,
        fanout: int,
        max_hops: int,
        all_pub_keys: dict[str, bytes],
        crypto_scheme: str,
    ):
        self.fanout = fanout
        self.max_hops = max_hops
        self.all_pub_keys = all_pub_keys
        self.crypto_scheme = crypto_scheme

        # track forwarded state: (origin_client_id, forwarder_client_id)
        self._seen_forward: set[tuple[str, str]] = set()
        self.gossip_timings: list[dict] = []

    def reset_round(self):
        self._seen_forward.clear()
        self.gossip_timings.clear()
        logging.info("Gossip round state reset")

    def _compute_expected_payload(self, message: dict) -> bytes:
        if "is_hashed" not in message:
            raise KeyError("Message missing required field: 'is_hashed'")

        if not message["is_hashed"]:
            return message["update_bytes"]

        if "hash_algorithm" not in message:
            raise KeyError("Message missing required field: 'hash_algorithm'")

        algo = message["hash_algorithm"].lower()

        if algo == "sha256":
            return hashlib.sha256(message["update_bytes"]).digest()

        raise ValueError(f"Unsupported hash algorithm: {algo}")

    def _verify_before_forward(self, message: dict) -> tuple[bool, float]:
        pk = self.all_pub_keys.get(message["client_id"])
        if pk is None:
            logging.error(f"Missing public key for {message['client_id']}")
            return False, 0.0

        expected_payload = self._compute_expected_payload(message)

        if expected_payload != message["payload"]:
            logging.warning(
                f"Payload mismatch detected for {message['client_id']} before forwarding"
            )
            return False, 0.0

        is_valid, verify_ms = dilithium_utils.verify(
            pk,
            message["payload"],
            message["signature"],
            self.crypto_scheme,
        )

        if not is_valid:
            logging.warning(f"Signature verification failed for {message['client_id']}")

        return is_valid, verify_ms

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
            is_valid, verify_ms = self._verify_before_forward(message)

            self.gossip_timings.append({
                "from": origin_node.client_id,
                "to": target.client_id,
                "origin": origin_client_id,
                "hop": hop + 1,
                "verify_ms": round(verify_ms, 3),
                "accepted": is_valid,
            })

            logging.info(
                f"[gossip] {origin_node.client_id} -> {target.client_id} "
                f"hop={hop + 1} verify={verify_ms:.3f} ms "
                f"[{'OK' if is_valid else 'REJECTED'}]"
            )

            if not is_valid:
                continue

            target.receive_gossip(message)
            self.spread(target, all_nodes, message, hop=hop + 1)

    def run_round(self, nodes):
        self.reset_round()

        for node in nodes:
            if node.own_submission is None:
                raise RuntimeError(
                    f"{node.client_id} has no submission — call sign_update() first"
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
            logging.info("No gossip timings recorded for this round")
            return

        logging.info("-" * 70)
        logging.info(f"Gossip log (fanout={self.fanout} max_hops={self.max_hops})")
        logging.info("-" * 70)
        logging.info(
            f"{'Origin':<12} {'From':<12} {'To':<12} {'Hop':<5} {'Verify (ms)':<14} Accepted"
        )
        logging.info("-" * 70)

        for t in self.gossip_timings:
            logging.info(
                f"{t['origin']:<12} {t['from']:<12} {t['to']:<12} {t['hop']:<5} "
                f"{t['verify_ms']:<14} {t['accepted']}"
            )

        accepted = [t for t in self.gossip_timings if t["accepted"]]
        if accepted:
            avg_v = sum(t["verify_ms"] for t in accepted) / len(accepted)
            logging.info(f"Total gossip hops: {len(self.gossip_timings)}")
            logging.info(f"Avg gossip verify: {avg_v:.3f} ms")