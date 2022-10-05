import pickle
import pandas as pd
import numpy as np


def read_pickle(filepath: str):
    with open(filepath, "rb") as file:
        output_file = pickle.load(file)
    return output_file


def write_pickle(filepath: str):
    with open(filepath, "wb") as file:
        pickle.dump(file)
