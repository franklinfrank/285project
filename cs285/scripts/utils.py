import itertools

from typing import List, Dict

#seeds = [285, 2020, 2, 423, 389, 147]
seeds = [285, 2020, 2]
#environments = ["CartPole-v0", "InvertedPendulum-v2"]
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

DEFAULTS = {
    "HalfCheetah-v2": {
        "terminal_val": 2,
        "sample_strategy": "ordered_random",
        "batch_size": 4000,
    },
    "CartPole-v0": {
        "terminal_val": 2,
        "sample_strategy": "sequential",
        "batch_size": 1000,
    },
    "InvertedPendulum-v2": {
        "terminal_val": 5,
        "sample_strategy": "sequential",
        "batch_size": 1000,
    },
}


def dict_product(dicts, *, master_key="env_name") -> List[Dict]:
    """
    Cross product of two dictionaries. Fills in default based on master key.
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    ret = list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
    for dct in ret:
        for k, v in dct.items():
            if v is None:
                dct[k] = DEFAULTS[dct[master_key]][k]
    return ret

def get_bl_name(exp_flags: Dict) -> str:
    return "bl" + "_".join([f"{k}={v}" for k, v in exp_flags.items()])

def get_exp_name(exp_flags: Dict) -> str:
    return "_".join([f"{k}={v}" for k, v in exp_flags.items()])


def _get_env_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=environments,
            sample_strategy=[None],
            terminal_val=[None],
            batch_size=[None],
        )
    )


def _get_batch_size_sweep() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=[DEFAULT_ENV],
            sample_strategy=[None],
            terminal_val=[None],
            batch_size=batch_sizes,
        )
    )


def _get_terminal_val_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=environments,
            sample_strategy=[None],
            terminal_val=terminal_values,
            batch_size=[None],
        )
    )


def _get_sample_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=environments,
            sample_strategy=strategies,
            terminal_val=[None],
            batch_size=[None],
        )
    )


def _get_sample_and_term_sweep_flags() -> List[Dict]:
    return dict_product(
        dict(
            seed=seeds,
            env_name=[DEFAULT_ENV],
            sample_strategy=strategies,
            terminal_val=terminal_values,
            batch_size=[None],
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
    elif exp_set == "sample_and_term_sweep":
        return _get_sample_and_term_sweep_flags()
    else:
        raise ValueError("Experiment set unrecognized!")
