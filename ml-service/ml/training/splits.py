"""Reproducible dataset splitting."""

from sklearn.model_selection import train_test_split


def split_dataset(
    X,
    y,
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
):
    """Split features and labels into train and test partitions.

    Stratification preserves class proportions when possible.
    Federated IID and non-IID client partitioning is implemented
    separately in the federated module.
    """
    if not 0.05 <= test_size <= 0.5:
        raise ValueError(
            "test_size must be between 0.05 and 0.5"
        )
    stratify_values = None
    if stratify and y.value_counts().min() >= 2:
        stratify_values = y
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_values,
    )
