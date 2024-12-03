# tests/test_utils.py

import pytest
from utils import allowed_file

def test_allowed_file():
    assert allowed_file("data.csv") == True
    assert allowed_file("data.xlsx") == True
    assert allowed_file("data.json") == True
    assert allowed_file("data.parquet") == True
    assert allowed_file("data.txt") == False
    assert allowed_file("data.pdf") == False
    assert allowed_file("data") == False
    assert allowed_file(".csv") == False
