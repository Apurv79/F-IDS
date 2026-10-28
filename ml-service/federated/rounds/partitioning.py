"""Partitioning strategies for federated learning (IID and Non-IID)."""

from collections import defaultdict
import numpy as np
import pandas as pd


def create_iid_partitions(
    frame: pd.DataFrame,
    num_clients: int = 3,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Create Independent and Identically Distributed (IID) client partitions.

    Each client receives an equal, randomized slice of data containing roughly
    the same proportion of every attack class.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    chunk_size = int(np.ceil(n / num_clients))
    return [
        shuffled.iloc[i * chunk_size : min((i + 1) * chunk_size, n)].reset_index(drop=True)
        for i in range(num_clients)
    ]


def create_non_iid_partitions(
    frame: pd.DataFrame,
    num_clients: int = 3,
    label_column: str = "label",
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Create Non-IID (heterogeneous) client partitions.

    In the real world, different enterprise branches experience completely
    different attacks (e.g. Branch A sees mostly SQL Injection, Branch B sees XSS).
    This function groups by attack class and sorts/skews distribution across clients.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    rng = np.random.RandomState(seed)
    client_shards: dict[int, list[pd.DataFrame]] = defaultdict(list)

    # Sort each class separately and partition into shards
    unique_labels = frame[label_column].unique()
    for label in unique_labels:
        subset = frame[frame[label_column] == label].sample(
            frac=1.0, random_state=seed
        )
        # Split into shards and allocate to clients with Dirichlet/skewed assignment
        subset_len = len(subset)
        shard_size = int(np.ceil(subset_len / num_clients))
        shards = [
            subset.iloc[i * shard_size : min((i + 1) * shard_size, subset_len)]
            for i in range(num_clients)
        ]
        perm = rng.permutation(num_clients)
        for client_idx, shard in zip(perm, shards, strict=False):
            if not shard.empty:
                client_shards[client_idx].append(shard)

    partitions: list[pd.DataFrame] = []
    for client_idx in range(num_clients):
        if client_shards[client_idx]:
            client_df = pd.concat(client_shards[client_idx], ignore_index=True)
            partitions.append(client_df.sample(frac=1.0, random_state=seed).reset_index(drop=True))
        else:
            partitions.append(pd.DataFrame(columns=frame.columns))

    return partitions
