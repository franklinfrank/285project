import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def sec1():
	pong_folder = "dqn_q1_PongNoFrameskip-v4_13-10-2019_16-37-02"
	pong_event = "events.out.tfevents.1571009822.shark2"
	fpath = os.path.join(pong_folder, pong_event)
	pong_path = filepath(fpath)
	pong_avgs = parse_avg_from_events(pong_path)
	num_iter = len(pong_avgs)
	plt.plot(np.arange(num_iter), pong_avgs)
	plt.title("Q1: DQN Performance on Pong")
	plt.xlabel("Time steps")
	plt.ylabel("Average Return")
	plt.savefig("q1_dqn_pong.png")
	plt.clf()
	
def sec3():
	sb_foldernames = ["pg_sb_no_rtg_dsa_CartPole-v0_24-09-2019_23-53-14",
	                  "pg_sb_rtg_dsa_CartPole-v0_24-09-2019_23-57-09",
	                  "pg_sb_rtg_na_CartPole-v0_24-09-2019_23-59-00"]
	sb_filenames = ["events.out.tfevents.1569394394.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569394629.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569394740.jeffrey-ThinkPad-T450s"]
	sb_paths = []
	
	for folder, event_file in zip(sb_foldernames, sb_filenames):
		fpath = os.path.join(folder, event_file)
		sb_paths.append(filepath(fpath))
	no_rtg_dsa_sb = parse_avg_from_events(sb_paths[0])
	rtg_dsa_sb = parse_avg_from_events(sb_paths[1])
	rtg_na_sb = parse_avg_from_events(sb_paths[2])
	plt.plot(np.arange(100), no_rtg_dsa_sb, label="Non-standardized advantages")
	plt.plot(np.arange(100), rtg_dsa_sb, label="Reward-to-go, non-standardized advantages")
	plt.plot(np.arange(100), rtg_na_sb, label="Reward-to-go, standardized advantages")
	plt.legend()
	plt.title("P3. Cartpole Results, Small Batch Size")
	plt.xlabel("Iteration")
	plt.ylabel("Average Return")
	plt.savefig("p3_cartpole_sb.png")
	plt.clf()
	lb_foldernames = ["pg_lb_no_rtg_dsa_CartPole-v0_25-09-2019_00-01-03",
	                  "pg_lb_rtg_dsa_CartPole-v0_25-09-2019_00-05-45",
	                  "pg_lb_rtg_na_CartPole-v0_25-09-2019_00-14-29"]
	lb_filenames = ["events.out.tfevents.1569394863.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569395145.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569395669.jeffrey-ThinkPad-T450s"]
	lb_paths = []
	for folder, event_file in zip(lb_foldernames, lb_filenames):
		fpath = os.path.join(folder, event_file)
		lb_paths.append(filepath(fpath))
	no_rtg_dsa_lb = parse_avg_from_events(lb_paths[0])
	rtg_dsa_lb = parse_avg_from_events(lb_paths[1])
	rtg_na_lb = parse_avg_from_events(lb_paths[2])
	plt.plot(np.arange(100), no_rtg_dsa_lb, label="Non-standardized advantages")
	plt.plot(np.arange(100), rtg_dsa_lb, label="Reward-to-go, non-standardized advantages")
	plt.plot(np.arange(100), rtg_na_lb, label="Reward-to-go, standardized advantages")
	plt.legend()
	plt.title("P3. Cartpole Results, Large Batch Size")
	plt.xlabel("Iteration")
	plt.ylabel("Average Return")
	plt.savefig("p3_cartpole_lb.png")



def sec4():
	folder = "pg_ip_b500_r.005_InvertedPendulum-v2_27-09-2019_16-59-46"
	efile = "events.out.tfevents.1569628786.jeffrey-ThinkPad-T450s"
	fpath = os.path.join(folder, efile)
	full_path = filepath(fpath)
	returns = parse_avg_from_events(full_path)
	plt.plot(np.arange(100), returns)
	plt.title("P4. InvertedPendulum with Batch Size 500, Learning Rate .005")
	plt.xlabel("Iteration")
	plt.ylabel("AverageReturn")
	plt.savefig("p4_ip.png")
def sec6():
	#NOTE: Need to rerun
	folder = "pg_ll_b40000_r0.005_LunarLanderContinuous-v2_25-09-2019_01-52-51"
	efile = "events.out.tfevents.1569401571.jeffrey-ThinkPad-T450s"
	fpath = os.path.join(folder, efile)
	full_path = filepath(fpath)
	returns = parse_avg_from_events(full_path)
	plt.plot(np.arange(100), returns)
	plt.title("P6. LunarLander")
	plt.xlabel("Iteration")
	plt.ylabel("AverageReturn")
	plt.savefig("p6_ll.png")
	last_ten = returns[-10:]
	print("Average of last ten: {}".format(sum(last_ten) / len(last_ten)))
	
def sec7():
	sweep_folders = ["pg_hc_b10000_lr0.005_nnbaseline_HalfCheetah-v2_26-09-2019_01-03-21",
	                 "pg_hc_b10000_lr0.01_nnbaseline_HalfCheetah-v2_26-09-2019_03-31-08",
	                 "pg_hc_b10000_lr0.02_nnbaseline_HalfCheetah-v2_26-09-2019_05-58-11",
	                 "pg_hc_b30000_lr0.005_nnbaseline_HalfCheetah-v2_26-09-2019_01-20-07",
	                 "pg_hc_b30000_lr0.01_nnbaseline_HalfCheetah-v2_26-09-2019_03-47-59",
	                 "pg_hc_b30000_lr0.02_nnbaseline_HalfCheetah-v2_26-09-2019_06-14-56",
	                 "pg_hc_b50000_lr0.005_nnbaseline_HalfCheetah-v2_26-09-2019_02-09-02",
	                 "pg_hc_b50000_lr0.01_nnbaseline_HalfCheetah-v2_26-09-2019_04-36-56",
	                 "pg_hc_b50000_lr0.02_nnbaseline_HalfCheetah-v2_26-09-2019_07-04-17"]
	sweep_events = ["events.out.tfevents.1569485001.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569493868.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569502691.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569486007.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569494879.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569503696.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569488942.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569497816.jeffrey-ThinkPad-T450s",
	                "events.out.tfevents.1569506657.jeffrey-ThinkPad-T450s"]
	sweep_paths = []
	for folder, event in zip(sweep_folders, sweep_events):
		sweep_path = os.path.join(folder, event)
		sweep_paths.append(sweep_path)
	i = 0
	for bsize in [10000, 30000, 50000]:
		for lr in [0.005, 0.01, 0.02]:
			full_path = filepath(sweep_paths[i])
			returns = parse_avg_from_events(full_path)
			plt.plot(np.arange(100), returns, label="Batch {}, LR {}".format(bsize, lr))
			i += 1
	plt.xlabel("Iterations")
	plt.ylabel("AverageReturn")
	plt.legend()
	plt.title("P7. HalfCheetah Performance Across Different Hyperparameters")
	plt.savefig("p7_sweep.png")
	plt.clf()

	spec_folders = ["pg_hc_b30000_lr.02_HalfCheetah-v2_26-09-2019_09-39-55",
	                "pg_hc_b30000_lr.02_HalfCheetah-v2_26-09-2019_15-13-56",
	                "pg_hc_b30000_lr0.02_HalfCheetah-v2_27-09-2019_21-32-28",
	                "pg_hc_b30000_lr0.02_HalfCheetah-v2_26-09-2019_06-14-56"]


	spec_events = ["events.out.tfevents.1569515995.jeffrey-ThinkPad-T450s",
	               "events.out.tfevents.1569536036.jeffrey-ThinkPad-T450s",
	               "events.out.tfevents.1569645148.jeffrey-ThinkPad-T450s",
	               "events.out.tfevents.1569503696.jeffrey-ThinkPad-T450s"]
	spec_paths = []
	for folder, event in zip(spec_folders, spec_events):
		spec_path = os.path.join(folder, event)
		spec_paths.append(filepath(spec_path))
	no_rtg_no_bl = parse_avg_from_events(spec_paths[0])
	rtg_no_bl = parse_avg_from_events(spec_paths[1])
	no_rtg_bl = parse_avg_from_events(spec_paths[2])
	print(no_rtg_bl)

	rtg_bl = parse_avg_from_events(spec_paths[3])
	print(rtg_bl)
	plt.plot(np.arange(100), no_rtg_no_bl, label="No RTG, no baseline")
	plt.plot(np.arange(100), rtg_no_bl, label="With RTG, no baseline")
	plt.plot(np.arange(100), no_rtg_bl, label="No RTG, with baseline")
	plt.plot(np.arange(100), rtg_bl, label="With RTG, with baseline")
	plt.legend()
	plt.xlabel("Iterations")
	plt.ylabel("AverageReturn")
	plt.title("P7. HalfCheetah with Batch Size 30000, Learning Rate .02")
	plt.savefig("p7_spec.png")

	

def parse_avg_from_events(filepath):
	eval_returns = []
	for e in tf.train.summary_iterator(filepath):
		for v in e.summary.value:
			if v.tag == 'Eval_AverageReturn':
				eval_returns.append(v.simple_value)
	return eval_returns

def filepath(filename):
	dir = os.path.dirname(__file__)
	return os.path.join(dir, "run_logs", filename)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=int, required=True, help="choices: 1, 2, 3, 4, 5")
    args = parser.parse_args()
    params = vars(args)
    if params["question"] == 1:
        sec1()
    elif params["question"] == 2:
        sec2()
    elif params["question"] == 3:
        sec3()
    elif params["question"] == 4:
    	sec4()
    elif params["question"] == 5:
    	sec5()
    else:
        print("Invalid choice, choices are 1-5")


if __name__ == "__main__":
	main()


