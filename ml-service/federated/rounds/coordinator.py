"""Federated round coordination, version validation, and simulation orchestration."""

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

import pandas as pd

from sklearn.model_selection import train_test_split

from federated.aggregation.fedavg import aggregate_parameters
from federated.clients.simulated_client import SimulatedClient
from federated.models.parameters import set_parameters
from federated.rounds.partitioning import (
    create_iid_partitions,
    create_non_iid_partitions,
)
from ml.evaluation.metrics import evaluate_classifier
from ml.features.extractor import extract_frame
from ml.models.pipeline import build_training_pipeline
from ml.training.splits import split_dataset


@dataclass
class FederatedRound:
    """State for one local simulation round."""

    round_id: str = field(default_factory=lambda: str(uuid4()))
    base_model_version: str = "v1"
    status: str = "pending"
    accepted_updates: list[dict] = field(default_factory=list)


class FederatedCoordinator:
    """Coordinate version-checked client updates and FedAvg aggregation."""

    def __init__(self, initial_model_version: str = "v1"):
        self.current_model_version = initial_model_version
        self.current_round: FederatedRound | None = None
        self.global_parameters: list[list[float]] | None = None

    def start_round(self) -> FederatedRound:
        """Start a round pinned to the current global model version."""
        if self.current_round and self.current_round.status in {
            "pending",
            "running",
            "aggregating",
        }:
            raise RuntimeError("A federated round is already active")
        self.current_round = FederatedRound(
            base_model_version=self.current_model_version,
            status="running",
        )
        return self.current_round

    def submit_update(
        self,
        client_id: str,
        base_model_version: str,
        parameters: list[list[float]],
        num_examples: int,
        metrics: dict | None = None,
    ) -> None:
        """Submit and validate a local client update for the active round."""
        if not self.current_round or self.current_round.status != "running":
            raise RuntimeError("No active running federated round")

        if base_model_version != self.current_round.base_model_version:
            raise ValueError(
                f"Stale model update from client {client_id}: "
                f"expected base version '{self.current_round.base_model_version}', "
                f"got '{base_model_version}'"
            )

        if num_examples <= 0:
            raise ValueError("num_examples must be greater than zero")

        self.current_round.accepted_updates.append({
            "client_id": client_id,
            "parameters": parameters,
            "num_examples": num_examples,
            "metrics": metrics or {},
        })

    def aggregate(self) -> dict:
        """Aggregate all accepted updates using FedAvg."""
        if not self.current_round or not self.current_round.accepted_updates:
            raise RuntimeError("No updates available to aggregate")

        self.current_round.status = "aggregating"

        updates_to_aggregate = [
            (update["parameters"], update["num_examples"])
            for update in self.current_round.accepted_updates
        ]

        aggregated = aggregate_parameters(updates_to_aggregate)
        self.global_parameters = aggregated

        previous_version = self.current_model_version
        new_version = f"{previous_version}-next"
        self.current_model_version = new_version
        self.current_round.status = "completed"

        client_count = len(self.current_round.accepted_updates)

        return {
            "round_id": self.current_round.round_id,
            "clients_aggregated": client_count,
            "previous_model_version": previous_version,
            "new_model_version": new_version,
            "aggregated_parameters": aggregated,
        }

    def run_simulation(
        self,
        dataset: pd.DataFrame,
        num_clients: int = 3,
        partition_strategy: Literal["iid", "non_iid"] = "iid",
        label_column: str = "label",
        random_seed: int = 42,
    ) -> dict:
        """Run a complete end-to-end multi-client simulation round."""
        # 1. Split global evaluation set vs training set
        train_data, test_data = train_test_split(
            dataset,
            test_size=0.2,
            random_state=random_seed,
            stratify=dataset[label_column] if dataset[label_column].value_counts().min() >= 2 else None,
        )

        # 2. Partition training data across edge clients
        if partition_strategy == "iid":
            client_datasets = create_iid_partitions(
                train_data, num_clients=num_clients, seed=random_seed
            )
        else:
            client_datasets = create_non_iid_partitions(
                train_data,
                num_clients=num_clients,
                label_column=label_column,
                seed=random_seed,
            )

        # 3. Start a round
        round_info = self.start_round()

        # 4. Each client trains on local partition
        for i, client_df in enumerate(client_datasets):
            client_id = f"client_{i + 1}"
            client = SimulatedClient(client_id=client_id, random_seed=random_seed + i)

            X_client = extract_frame(client_df)
            y_client = client_df[label_column]

            update = client.fit(
                X_client,
                y_client,
                base_parameters=self.global_parameters,
            )

            self.submit_update(
                client_id=client_id,
                base_model_version=round_info.base_model_version,
                parameters=update["parameters"],
                num_examples=update["num_examples"],
            )

        # 5. Aggregate with FedAvg
        aggregation_result = self.aggregate()

        # 6. Evaluate aggregated global model on test set
        X_test = extract_frame(test_data)
        y_test = test_data[label_column]

        eval_pipeline = build_training_pipeline("logistic_regression", random_seed=random_seed)
        eval_pipeline.fit(extract_frame(train_data), train_data[label_column])

        # Apply the FedAvg parameters to the global classifier
        classifier = eval_pipeline.named_steps["classifier"]
        set_parameters(classifier, aggregation_result["aggregated_parameters"])

        eval_metrics = evaluate_classifier(eval_pipeline, X_test, y_test)

        return {
            "round_id": aggregation_result["round_id"],
            "strategy": partition_strategy,
            "num_clients": num_clients,
            "previous_model_version": aggregation_result["previous_model_version"],
            "new_model_version": aggregation_result["new_model_version"],
            "test_accuracy": eval_metrics["accuracy"],
            "test_f1_macro": eval_metrics["f1_macro"],
            "metrics": eval_metrics,
        }
