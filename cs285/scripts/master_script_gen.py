import itertools

seeds = [285, 2020, 2]

# Default arg first
environments = ['HalfCheetah-v2', 'Cartpole-v0', 'InvertedPendulum-v2']
strategies = ['ordered_random', 'pure_random', 'sequential', 'constrained_random', 'mixed']
terminal_values = ['learn', 0, 2, 5, 10, 50]
batch_size = [1000, 2000, 3000, 4000]

DEFAULT_ENV = environments[0]
DEFAULT_STRAT = strategies[0]
DEFAULT_TERM = terminal_values[2]
DEFAULT_BATCH_SIZE = batch_size[0]

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

py_cmd = "python run_hw3_actor_critic.py --env_name {env_name} "\
    "--ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 "\
    "-l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 "\
    "-diff --num_critic_updates_per_agent_update 10 --exp_name {name} "\
    "--seed {seed} --sample_strategy {sample_strategy} "\
    "--terminal_val {terminal_val} -tb {batch_size} -gpu"

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
        name = '_'.join([f'{k}={v}' for k, v in exp_flags.items()])
        exp_flags['name'] = name
        exp_cmd = py_cmd.format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

