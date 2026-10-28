"""Federated Averaging (FedAvg) for compatible numeric parameters."""

import numpy as np

from federated.models.parameters import validate_parameters


def aggregate_parameters(
    client_updates: list[tuple[list[list[float]], int]],
) -> list[list[float]]:
    """Aggregate client parameters using example-count weights.

    Each item is (parameters, number_of_training_examples).
    This function only averages numeric arrays. It does not validate
    client identity, authenticate submissions, or prove updates are honest.
    Those responsibilities belong to the round coordinator and transport.
    """
    if not client_updates:
        raise ValueError("At least one client update is required")

    total_examples = 0
    weighted_arrays = None
    expected_shapes = None

    for parameters, num_examples in client_updates:
        if num_examples <= 0:
            raise ValueError("Client example count must be positive")
        validate_parameters(parameters)

        arrays = [
            np.asarray(parameter, dtype=float)
            for parameter in parameters
        ]
        shapes = [array.shape for array in arrays]

        if expected_shapes is None:
            expected_shapes = shapes
            weighted_arrays = [
                np.zeros(shape, dtype=float)
                for shape in shapes
            ]
        elif shapes != expected_shapes:
            raise ValueError("Client parameter shapes do not match")

        weight = num_examples
        for index, array in enumerate(arrays):
            weighted_arrays[index] += array * weight
        total_examples += num_examples

    averaged = [
        (array / total_examples).tolist()
        for array in weighted_arrays
    ]
    return averaged
