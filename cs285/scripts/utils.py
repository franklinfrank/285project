from typing import Dict

def get_exp_name(exp_flags: Dict) -> str:
    return '_'.join([f'{k}={v}' for k, v in exp_flags.items()])

