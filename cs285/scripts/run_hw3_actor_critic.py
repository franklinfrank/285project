import os
import gym
import pdb
import time
import numpy as np 
import tensorflow as tf 

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.ac_agent import ACAgent

class AC_Trainer(object):

    def __init__(self, params):
    
        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'critic_learning_rate': params['critic_learning_rate'],
            'num_target_updates': params['num_target_updates'],
            'num_grad_steps_per_target_update': params['num_grad_steps_per_target_update'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'differential': params['differential'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
            'terminal_val': params['terminal_val'],
            'sample_strategy': params['sample_strategy'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = ACAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'], 
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)

    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=4000) ##steps used per gradient step

    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--critic_learning_rate', '-clr', type=float, default=5e-3)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--differential', '-diff', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--sample_strategy', '-sstrat', type=str, default='ordered_random', choices=['ordered_random', 'pure_random', 'sequential', 'constrained_random', 'mixed'])
    parser.add_argument('--terminal_val', '-tval', type=str, default='2', choices=['0', '2', '5', '10', '50', 'learn'])

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1) 
    parser.add_argument('--scalar_log_freq', type=int, default=10)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args() 

    # convert to dictionary
    params = vars(args)

    # for policy gradient, we made a design decision
    # to force batch_size = train_batch_size
    # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'ac_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = AC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
