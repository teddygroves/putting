"""Provides functions prepare_data_x.

These functions should take in a dataframe of measurements and return a
PreparedData object.

Note that you can change the input arbitrarily - for example if you want to take
in two dataframes, a dictionary etc. However in this case you will need to edit
the corresponding code in the file prepare_data.py accordingly.

"""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import KFold

from src.prepared_data import PreparedData
from src.util import CoordDict, StanInput, check_is_df, stanify_dict

HARDCODED_NUMBERS = {
    "r": (1.68 / 2) / 12,  # radius of a 1.68 inch wide golf ball in feet
    "R": (4.25 / 2) / 12,  # radius of a 4.25 inch wide golf hole in feet
    "overshot": 1,  # putter's target distance minus x
    "distance_tolerance": 3,  # success only if distance is in [x, x+3]
}

NEW_COLNAMES = {"yButIThoughtIdAddSomeLetters": "y"}
N_CV_FOLDS = 10
DIMS = {
    "y": ["observation"],
    "batch_size": ["observation"],
    "yrep": ["observation"],
    "llik": ["observation"],
}


def prepare_data_old(measurements_raw: pd.DataFrame) -> PreparedData:
    """Prepare the old data."""
    measurements = process_measurements(measurements_raw, "old")
    return PreparedData(
        name="old",
        coords=CoordDict({"observation": measurements.index.tolist()}),
        dims=DIMS,
        measurements=measurements,
        number_of_cv_folds=N_CV_FOLDS,
        stan_input_function=get_stan_input,
    )


def prepare_data_new(measurements_raw: pd.DataFrame) -> PreparedData:
    """Prepare the new data."""
    measurements = process_measurements(measurements_raw, "new")
    return PreparedData(
        name="new",
        coords=CoordDict({"observation": measurements.index.tolist()}),
        dims=DIMS,
        measurements=measurements,
        number_of_cv_folds=N_CV_FOLDS,
        stan_input_function=get_stan_input,
    )


def process_measurements(measurements_raw: pd.DataFrame, pref) -> pd.DataFrame:
    """Process the measurements.

    Since the data is already as clean as possible, this function just adds a
    prefix to the index.

    """
    return check_is_df(measurements_raw.copy().rename(lambda i: f"{pref}_{i}"))


def get_stan_input(
    measurements: pd.DataFrame,
    likelihood: bool,
    train_ix: List[int],
    test_ix: List[int],
) -> StanInput:
    """Turn a processed dataframe into a Stan input."""
    return stanify_dict(
        {
            "N": len(measurements),
            "N_train": len(train_ix),
            "N_test": len(test_ix),
            "x": measurements["x"],
            "y": measurements["y"],
            "batch_size": measurements["n"],
            "likelihood": int(likelihood),
            "ix_train": [i + 1 for i in train_ix],
            "ix_test": [i + 1 for i in test_ix],
            "r": HARDCODED_NUMBERS["r"],
            "R": HARDCODED_NUMBERS["R"],
            "overshot": HARDCODED_NUMBERS["overshot"],
            "distance_tolerance": HARDCODED_NUMBERS["distance_tolerance"],
        }
    )


def get_stan_inputs(
    prepared_data: PreparedData,
) -> Tuple[StanInput, StanInput, List[StanInput]]:
    """Get Stan input dictionaries for all modes from a PreparedData object."""
    ix_all = list(range(len(prepared_data.measurements)))
    stan_input_prior, stan_input_posterior = (
        prepared_data.stan_input_function(
            measurements=prepared_data.measurements,
            train_ix=ix_all,
            test_ix=ix_all,
            likelihood=likelihood,
        )
        for likelihood in (False, True)
    )
    stan_inputs_cv = []
    kf = KFold(prepared_data.number_of_cv_folds, shuffle=True)
    for train, test in kf.split(prepared_data.measurements):
        stan_inputs_cv.append(
            prepared_data.stan_input_function(
                measurements=prepared_data.measurements,
                likelihood=True,
                train_ix=list(train),
                test_ix=list(test),
            )
        )
    return stan_input_prior, stan_input_posterior, stan_inputs_cv
