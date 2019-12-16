tmux     new-session -s seed:285_env_name:HalfCheetah-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:285_env_name:HalfCheetah-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 285 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:285_env_name:Cartpole-v0_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name Cartpole-v0 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:285_env_name:Cartpole-v0_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 285 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:285_env_name:InvertedPendulum-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:285_env_name:InvertedPendulum-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 285 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:2020_env_name:HalfCheetah-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:2020_env_name:HalfCheetah-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 2020 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:2020_env_name:Cartpole-v0_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name Cartpole-v0 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:2020_env_name:Cartpole-v0_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 2020 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:2020_env_name:InvertedPendulum-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:2020_env_name:InvertedPendulum-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 2020 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:2_env_name:HalfCheetah-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:2_env_name:HalfCheetah-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 2 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:2_env_name:Cartpole-v0_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name Cartpole-v0 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:2_env_name:Cartpole-v0_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 2 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 
tmux     new-session -s seed:2_env_name:InvertedPendulum-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000  -eb 1500 -lr 0.02  -ntu 10 -ngsptu 10 -diff --num_critic_updates_per_agent_update 10 --exp_name seed:2_env_name:InvertedPendulum-v2_sample_strategy:ordered_random_terminal_val:2_batch_size:1000 --seed 2 --sample_strategy ordered_random --terminal_val 2 -tb 1000 -gpu'  \;     detach-client 