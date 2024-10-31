import pytest
import numpy as np
import json
from scripts.parse_json import (
    generate_features,
    cluster_samples,
    parse_row,
    process_line,
)


def test_generate_features():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = generate_features(array)

    assert len(result) == 15  # 5 sets of 3 elements (mean, median, max, min, sd)

    assert np.allclose(result[:3], [4, 5, 6])  # mean
    assert np.allclose(result[3:6], [4, 5, 6])  # median
    assert np.allclose(result[6:9], [7, 8, 9])  # max
    assert np.allclose(result[9:12], [1, 2, 3])  # min
    assert np.allclose(result[12:], [2.44948974, 2.44948974, 2.44948974])  # sd


def test_cluster_samples():
    array = np.array([[1, 2], [1, 1], [10, 10], [10, 11]])
    cluster_1, cluster_2 = cluster_samples(array)

    assert cluster_1.shape == (2, 2) and cluster_2.shape == (2, 2)
    assert np.array_equal(cluster_1, np.array([[1, 2], [1, 1]])) or np.array_equal(
        cluster_2, np.array([[1, 2], [1, 1]])
    )


def test_parse_row():
    mock_row = {
        "transcript_id_1": {
            "position_1": {"sequence_1": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        }
    }
    result = parse_row(mock_row)

    assert len(result) == 48


def test_process_line():
    mock_line = json.dumps(
        {
            "transcript_id_1": {
                "position_1": {"sequence_1": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
            }
        }
    )

    result = process_line(0, mock_line)

    assert len(result) == 48
