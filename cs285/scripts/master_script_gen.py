import itertools
from typing import Dict 

import utils

seeds = [285, 2020, 2]

# Default arg first
environments = ['HalfCheetah-v2', 'CartPole-v0', 'InvertedPendulum-v2']
strategies = ['ordered_random', 'pure_random', 'sequential', 'constrained_random', 'mixed']
terminal_values = ['learn', 0, 2, 5, 10, 50]
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

tmux_cmd = "tmux \
    new-session -s {name} '{py_cmd}'  \; \
    detach-client \n"

template_cmds = {
    'InvertedPendulum-v2': \
        "python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b "\
        "1000 --exp_name {name} -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed {seed} "\
        "--discount 0.95 -diff --sample_strategy {sample_strategy} "\
        "--terminal_val {terminal_val} -tb 1000",
    'HalfCheetah-v2': \
        "python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu "\
        "--ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 "\
        "-b 30000 -eb 1500 -lr 0.02 --exp_name {name} -ntu 10 "\
        "-ngsptu 10 -diff --seed {seed} --num_critic_updates_per_agent_update 10 "\
        "--sample_strategy {sample_strategy} --terminal_val {terminal_val} "\
        "-tb {batch_size} ",
    'CartPole-v0': \
        "python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 "\
        "--seed {seed} --exp_name {name} -ntu 10 -ngsptu 10 -diff -gpu "\
        "--sample_strategy {sample_strategy} --terminal_val {terminal_val} "\
        "-tb 1000 "
}


#####
#####
#####

# Search over environments
env_sweep_set = "run_env_sweep.sh"
with open(env_sweep_set, 'w+') as f:
    for exp_flags in dict_product(dict(seed=seeds, 
                                       env_name=environments, 
                                       sample_strategy=[DEFAULT_STRAT], 
                                       terminal_val=[DEFAULT_TERM], 
                                       batch_size=[DEFAULT_BATCH_SIZE])):
        name = utils.get_exp_name(exp_flags)
        exp_flags['name'] = name
        exp_cmd = template_cmds[exp_flags['env_name']].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

# Search over sampling strategies
sample_sweep_set = 'run_sample_sweep.sh'
with open(sample_sweep_set, 'w+') as f:
    for exp_flags in dict_product(dict(seed=seeds, 
                                       env_name=[DEFAULT_ENV], 
                                       sample_strategy=strategies, 
                                       terminal_val=[DEFAULT_TERM], 
                                       batch_size=[DEFAULT_BATCH_SIZE])):
        name = utils.get_exp_name(exp_flags)
        exp_flags['name'] = name
        exp_cmd = template_cmds[exp_flags['env_name']].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

# Search over terminal values
terminal_val_sweep_set = 'run_terminal_val_sweep.sh'
with open(terminal_val_sweep_set, 'w+') as f:
    for exp_flags in dict_product(dict(seed=seeds, 
                                       env_name=[DEFAULT_ENV], 
                                       sample_strategy=[DEFAULT_STRAT], 
                                       terminal_val=terminal_values, 
                                       batch_size=[DEFAULT_BATCH_SIZE])):
        name = utils.get_exp_name(exp_flags)
        exp_flags['name'] = name
        exp_cmd = template_cmds[exp_flags['env_name']].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

# Search over batch sizes
batch_size_sweep_set = 'run_batch_size_sweep.sh'
with open(batch_size_sweep_set, 'w+') as f:
    for exp_flags in dict_product(dict(seed=seeds, 
                                       env_name=[DEFAULT_ENV], 
                                       sample_strategy=[DEFAULT_STRAT], 
                                       terminal_val=[DEFAULT_TERM], 
                                       batch_size=batch_sizes)):
        name = utils.get_exp_name(exp_flags)
        exp_flags['name'] = name
        exp_cmd = template_cmds[exp_flags['env_name']].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)
