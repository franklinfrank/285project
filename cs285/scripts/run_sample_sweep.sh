tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy ordered_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy pure_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy constrained_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=285_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=285_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 285 --discount 0.95 -diff --sample_strategy mixed --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy ordered_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy pure_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy constrained_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2020_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2020_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2020 --discount 0.95 -diff --sample_strategy mixed --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy ordered_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy pure_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy constrained_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=2_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=2_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff --sample_strategy mixed --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=423_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=423_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 423 --discount 0.95 -diff --sample_strategy ordered_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=423_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=423_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 423 --discount 0.95 -diff --sample_strategy pure_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=423_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=423_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 423 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=423_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=423_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 423 --discount 0.95 -diff --sample_strategy constrained_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=423_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=423_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 423 --discount 0.95 -diff --sample_strategy mixed --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=389_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=389_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 389 --discount 0.95 -diff --sample_strategy ordered_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=389_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=389_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 389 --discount 0.95 -diff --sample_strategy pure_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=389_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=389_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 389 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=389_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=389_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 389 --discount 0.95 -diff --sample_strategy constrained_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=389_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=389_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 389 --discount 0.95 -diff --sample_strategy mixed --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=147_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=147_env_name=InvertedPendulum-v2_sample_strategy=ordered_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 147 --discount 0.95 -diff --sample_strategy ordered_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=147_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=147_env_name=InvertedPendulum-v2_sample_strategy=pure_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 147 --discount 0.95 -diff --sample_strategy pure_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=147_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=147_env_name=InvertedPendulum-v2_sample_strategy=sequential_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 147 --discount 0.95 -diff --sample_strategy sequential --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=147_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=147_env_name=InvertedPendulum-v2_sample_strategy=constrained_random_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 147 --discount 0.95 -diff --sample_strategy constrained_random --terminal_val 5 -tb 1000'  \;     detach-client
tmux     new-session -s seed=147_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 'python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name seed=147_env_name=InvertedPendulum-v2_sample_strategy=mixed_terminal_val=5_batch_size=1000 -ntu 10 -ngsptu 10 -gpu -clr 5e-3 --seed 147 --discount 0.95 -diff --sample_strategy mixed --terminal_val 5 -tb 1000'  \;     detach-client
