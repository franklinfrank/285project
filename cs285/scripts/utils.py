import itertools

from typing import List, Dict

seeds = [285, 2020, 2]
environments = ["HalfCheetah-v2", "CartPole-v0", "InvertedPendulum-v2"]
strategies = [
    "ordered_random",
    "pure_random",
    "sequential",
    "constrained_random",
    "mixed",
]
terminal_values = ["learn", 0, 2, 5, 10, 50]
batch_sizes = [1000, 2000, 3000, 4000]

DEFAULT_ENV = environments[0]
DEFAULT_STRAT = strategies[0]
DEFAULT_TERM = terminal_values[2]
DEFAULT_BATCH_SIZE = batch_sizes[3]

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def get_exp_name(exp_flags: Dict) -> str:
    return "_".join([f"{k}={v}" for k, v in exp_flags.items()])


def _get_env_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=environments,
            sample_strategy=[DEFAULT_STRAT],
            terminal_val=[DEFAULT_TERM],
            batch_size=[DEFAULT_BATCH_SIZE],
        )
    )


def _get_batch_size_sweep() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=[DEFAULT_ENV],
            sample_strategy=[DEFAULT_STRAT],
            terminal_val=[DEFAULT_TERM],
            batch_size=batch_sizes,
        )
    )


def _get_terminal_val_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=[DEFAULT_ENV],
            sample_strategy=[DEFAULT_STRAT],
            terminal_val=terminal_values,
            batch_size=[DEFAULT_BATCH_SIZE],
        )
    )


def _get_sample_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=[DEFAULT_ENV],
            sample_strategy=strategies,
            terminal_val=[DEFAULT_TERM],
            batch_size=[DEFAULT_BATCH_SIZE],
        )
    )


def get_exp_flags(exp_set: str) -> List[Dict]:
    if exp_set == "env_sweep":
        return _get_env_sweep_flags()
    elif exp_set == "sample_sweep":
        return _get_sample_sweep_flags()
    elif exp_set == "terminal_val_sweep":
        return _get_terminal_val_sweep_flags()
    elif exp_set == "batch_size_sweep":
        return _get_batch_size_sweep()
    else:
        raise ValueError("Experiment set unrecognized!")