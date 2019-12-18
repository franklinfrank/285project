import itertools
from typing import Dict

import utils

tmux_cmd = "tmux \
    new-session -s {name} '{py_cmd}'  \; \
    detach-client \n"

template_cmds = {
    "InvertedPendulum-v2": "python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b "
    "1000 --exp_name {name} -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed {seed} "
    "--discount 0.95 -diff --sample_strategy {sample_strategy} "
    "--terminal_val {terminal_val} -tb 1000",
    "HalfCheetah-v2": "python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu "
    "--ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 "
    "-b 30000 -eb 1500 -lr 0.02 --exp_name {name} -ntu 10 "
    "-ngsptu 10 -diff --seed {seed} --num_critic_updates_per_agent_update 10 "
    "--sample_strategy {sample_strategy} --terminal_val {terminal_val} "
    "-tb {batch_size} ",
    "CartPole-v0": "python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 "
    "--seed {seed} --exp_name {name} -ntu 10 -ngsptu 10 -diff -gpu "
    "--sample_strategy {sample_strategy} --terminal_val {terminal_val} "
    "-tb 1000 ",
}


#####
#####
#####

# Search over environments
env_sweep_set = "run_env_sweep.sh"
with open(env_sweep_set, "w+") as f:
    for exp_flags in utils.get_exp_flags("env_sweep"):
        name = utils.get_exp_name(exp_flags)
        exp_flags["name"] = name
        exp_cmd = template_cmds[exp_flags["env_name"]].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

# Search over sampling strategies
sample_sweep_set = "run_sample_sweep.sh"
with open(sample_sweep_set, "w+") as f:
    for exp_flags in utils.get_exp_flags("sample_sweep"):
        name = utils.get_exp_name(exp_flags)
        exp_flags["name"] = name
        exp_cmd = template_cmds[exp_flags["env_name"]].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

# Search over terminal values
terminal_val_sweep_set = "run_terminal_val_sweep.sh"
with open(terminal_val_sweep_set, "w+") as f:
    for exp_flags in utils.get_exp_flags("terminal_val_sweep"):
        name = utils.get_exp_name(exp_flags)
        exp_flags["name"] = name
        exp_cmd = template_cmds[exp_flags["env_name"]].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)

# Search over batch sizes
batch_size_sweep_set = "run_batch_size_sweep.sh"
with open(batch_size_sweep_set, "w+") as f:
    for exp_flags in utils.get_exp_flags("batch_size_sweep"):
        name = utils.get_exp_name(exp_flags)
        exp_flags["name"] = name
        exp_cmd = template_cmds[exp_flags["env_name"]].format_map(exp_flags)
        exp_cmd = tmux_cmd.format(**(dict(name=name, py_cmd=exp_cmd)))
        f.write(exp_cmd)
