python run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 10_10 -ntu 10 -ngsptu 10 -diff -gpu

python run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name hcheet -ntu 10 -ngsptu 10 -diff -gpu 

python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 -n 100 -b 1000 --exp_name 10_10 -ntu 5 -ngsptu 5 -gpu -clr 5e-3 --seed 2 --discount 0.95 -diff -gpu 
