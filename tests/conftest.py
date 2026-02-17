"""Shared fixtures for PsychoBench unit tests."""
import os
import sys

# Add project root so "utils" and "example_generator" can be imported when running pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import pytest


@pytest.fixture
def questionnaire_avg():
    """Minimal questionnaire with compute_mode AVG, 3 questions, reverse [1], scale 5."""
    return {
        "name": "Minimal",
        "questions": {"1": "Q1", "2": "Q2", "3": "Q3"},
        "scale": 5,
        "compute_mode": "AVG",
        "reverse": [1],
        "categories": [
            {
                "cat_name": "Overall",
                "cat_questions": [1, 2, 3],
                "crowd": [{"crowd_name": "Crowd", "mean": 3.0, "std": 0.5, "n": 50}],
            }
        ],
    }


@pytest.fixture
def questionnaire_sum():
    """Minimal questionnaire with compute_mode SUM."""
    return {
        "name": "MinimalSum",
        "questions": {"1": "Q1", "2": "Q2", "3": "Q3"},
        "scale": 5,
        "compute_mode": "SUM",
        "reverse": [],
        "categories": [
            {
                "cat_name": "Overall",
                "cat_questions": [1, 2, 3],
                "crowd": [],
            }
        ],
    }


@pytest.fixture
def questionnaire_two_categories():
    """Questionnaire with two categories: [1,2] and [3]."""
    return {
        "name": "TwoCat",
        "questions": {"1": "Q1", "2": "Q2", "3": "Q3"},
        "scale": 5,
        "compute_mode": "AVG",
        "reverse": [],
        "categories": [
            {"cat_name": "A", "cat_questions": [1, 2], "crowd": []},
            {"cat_name": "B", "cat_questions": [3], "crowd": []},
        ],
    }


@pytest.fixture
def questionnaire_scale8_reverse1():
    """Questionnaire with scale 8 and reverse [1] for convert_data reverse-scaling tests."""
    return {
        "name": "Scale8",
        "questions": {"1": "Q1", "2": "Q2", "3": "Q3"},
        "scale": 8,
        "compute_mode": "AVG",
        "reverse": [1],
        "categories": [
            {"cat_name": "Overall", "cat_questions": [1, 2, 3], "crowd": []}
        ],
    }


@pytest.fixture
def questionnaire_no_reverse():
    """Questionnaire with empty reverse list."""
    return {
        "name": "NoReverse",
        "questions": {"1": "Q1", "2": "Q2", "3": "Q3"},
        "scale": 8,
        "compute_mode": "AVG",
        "reverse": [],
        "categories": [
            {"cat_name": "Overall", "cat_questions": [1, 2, 3], "crowd": []}
        ],
    }


def write_convert_data_csv(path, rows_by_columns):
    """
    Write a CSV for convert_data in the transposed format (each list is a column).
    rows_by_columns: list of lists; each inner list is one column's values in order.
    """
    rows = list(zip(*rows_by_columns))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


@pytest.fixture
def csv_one_order_one_test(tmp_path):
    """CSV with one order block and one response column. Q1=5 (reverse->3), Q2=4, Q3=4."""
    path = tmp_path / "test.csv"
    # Columns so first file row is (Prompt, order-0, shuffle0-test0) for convert_data
    write_convert_data_csv(
        path,
        [
            ["Prompt: minimal", "1. Q1", "2. Q2", "3. Q3"],
            ["order-0", "1", "2", "3"],
            ["shuffle0-test0", "5", "4", "4"],
        ],
    )
    return path


@pytest.fixture
def csv_no_reverse(tmp_path):
    """CSV with no reverse scaling; values stored as-is."""
    path = tmp_path / "test.csv"
    write_convert_data_csv(
        path,
        [
            ["Prompt: minimal", "1. Q1", "2. Q2", "3. Q3"],
            ["order-0", "1", "2", "3"],
            ["shuffle0-test0", "2", "3", "4"],
        ],
    )
    return path


@pytest.fixture
def csv_two_orders(tmp_path):
    """CSV with order-0 and order-1, each with one response column (6 columns, 4 rows)."""
    path = tmp_path / "test.csv"
    # Columns so first file row has order-0 and order-1 for convert_data
    # 6 columns so first file row is (Prompt, order-0, shuffle0-test0, Prompt, order-1, shuffle1-test0)
    write_convert_data_csv(
        path,
        [
            ["Prompt: x", "1. Q1", "2. Q2", "3. Q3"],
            ["order-0", "1", "2", "3"],
            ["shuffle0-test0", "5", "4", "4"],
            ["Prompt: x", "1. Q1", "2. Q2", "3. Q3"],
            ["order-1", "2", "3", "1"],
            ["shuffle1-test0", "2", "3", "1"],
        ],
    )
    return path


@pytest.fixture
def csv_invalid_cell(tmp_path):
    """CSV with a non-integer in a response cell."""
    path = tmp_path / "test.csv"
    write_convert_data_csv(
        path,
        [
            ["Prompt: x", "1. Q1", "2. Q2", "3. Q3"],
            ["order-0", "1", "2", "3"],
            ["shuffle0-test0", "x", "2", "3"],
        ],
    )
    return path
