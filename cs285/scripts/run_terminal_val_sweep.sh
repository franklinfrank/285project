tmux     new-session -s seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=learn_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=learn_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 285 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val learn -tb 4000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=0_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=0_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 285 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 0 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=2_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=2_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 285 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 2 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 285 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 5 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=10_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=10_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 285 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 10 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=50_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=285_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=50_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 285 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 50 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=learn_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 285 --exp_name seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=learn_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val learn -tb 1000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=0_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 285 --exp_name seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=0_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 0 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=2_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 285 --exp_name seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=2_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 2 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 285 --exp_name seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 5 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=10_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 285 --exp_name seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=10_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 10 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=50_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 285 --exp_name seed=285_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=50_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 50 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=learn_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=learn_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val learn -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=0_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=0_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val 0 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=2_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=2_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val 2 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=10_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=10_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val 10 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=50_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=50_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val 50 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=learn_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=learn_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2020 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val learn -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=0_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=0_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2020 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 0 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=2_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=2_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2020 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 2 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2020 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 5 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=10_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=10_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2020 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 10 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=50_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2020_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=50_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2020 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 50 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=learn_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2020 --exp_name seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=learn_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val learn -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=0_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2020 --exp_name seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=0_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 0 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=2_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2020 --exp_name seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=2_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 2 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2020 --exp_name seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 5 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=10_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2020 --exp_name seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=10_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 10 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=50_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2020 --exp_name seed=2020_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=50_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 50 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=learn_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=learn_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val learn -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=0_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=0_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val 0 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=2_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=2_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val 2 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=10_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=10_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val 10 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=50_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=50_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val 50 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=learn_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=learn_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val learn -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=0_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=0_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 0 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=2_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=2_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 2 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 5 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=10_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=10_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 10 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=50_batch_size=4000 'python run_hw3_actor_critic.py --env_name HalfCheetah-v2 -gpu --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name seed=2_env_name=HalfCheetah-v2_sample_strategy=ordered_random_terminal_val=50_batch_size=4000 -ntu 10 -ngsptu 10 -diff --seed 2 --num_critic_updates_per_agent_update 10 --sample_strategy ordered_random --terminal_val 50 -tb 4000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=learn_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2 --exp_name seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=learn_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val learn -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=0_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2 --exp_name seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=0_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 0 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=2_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2 --exp_name seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=2_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 2 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2 --exp_name seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 5 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=10_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2 --exp_name seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=10_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 10 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=50_batch_size=1000 'python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --seed 2 --exp_name seed=2_env_name=CartPole-v0_sample_strategy=sequential_terminal_val=50_batch_size=1000 -ntu 10 -ngsptu 10 -diff -gpu --sample_strategy sequential --terminal_val 50 -tb 1000 '  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=learn_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=learn_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val learn -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=0_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=0_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val 0 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=2_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=2_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val 2 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=10_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=10_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val 10 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=50_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=50_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val 50 -tb 1000'  \;     detach-client
