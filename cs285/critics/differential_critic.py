from .bootstrapped_continuous_critic import BootstrappedContinuousCritic
import tensorflow as tf
from cs285.infrastructure.tf_utils import build_mlp
import numpy as np

class DifferentialCritic(BootstrappedContinuousCritic):
    def __init__(self, sess, hparams):
        super().__init__(sess, hparams)
        self.terminal_val = hparams['terminal_val']
        self.sample_strat = hparams['sample_strategy']
    
    def _build(self):

        self.diff_sy_ob_no, self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()
        self.n_diff_layers = self.n_layers
        self.diff_size = self.size
        self.n_diff_layers = 4 
        self.diff_size = 128

        # define the critic
        self.diff_critic_prediction = tf.squeeze(build_mlp(
            self.diff_sy_ob_no,
            1,
            "nn_diff_critic",
            n_layers=self.n_diff_layers,
            size=self.diff_size,
            activation=tf.nn.relu))
        self.diff_sy_target_n = tf.placeholder(shape=[None], name="diff_critic_target", dtype=tf.float32)

       

        # TODO: set up the critic loss
        # HINT1: the critic_prediction should regress onto the targets placeholder (sy_target_n)
        # HINT2: use tf.losses.mean_squared_error
        self.diff_critic_loss = tf.losses.mean_squared_error(self.diff_critic_prediction, self.diff_sy_target_n)

        # TODO: use the AdamOptimizer to optimize the loss defined above
        self.diff_critic_update_op = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.diff_critic_loss)

        self.critic_prediction = tf.squeeze(build_mlp(
            self.sy_ob_no,
            1,
            "nn_critic",
            n_layers=self.n_layers,
            size=self.size))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)

        self.critic_loss = tf.losses.mean_squared_error(self.critic_prediction, self.sy_target_n)

        # TODO: use the AdamOptimizer to optimize the loss defined above
        self.critic_update_op = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.critic_loss)
 
    
    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        diff_sy_ob_no = tf.placeholder(shape=[None, 2*self.ob_dim], name="ob", dtype=tf.float32)
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        return diff_sy_ob_no, sy_ob_no, sy_ac_na, sy_adv_n

    def forward(self, ob_1, ob_2):
            # TODO: run your critic
            # HINT: there's a neural network structure defined above with mlp layers, which serves as your 'critic'
            ob = np.concatenate((ob_1, ob_2), axis=1)
            return self.sess.run(self.diff_critic_prediction, feed_dict={self.diff_sy_ob_no: ob})

    def single_forward(self, ob):
            # TODO: run your critic
            # HINT: there's a neural network structure defined above with mlp layers, which serves as your 'critic'
            return self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: ob})
    

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the sampled paths
            let num_paths be the number of sampled paths

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                loss
        """

        # TODO: Implement the pseudocode below: 

        # do the following (self.num_grad_steps_per_target_update * self.num_target_updates) times:
            # every self.num_grad_steps_per_target_update steps (which includes the first step),
                # recompute the target values by 
                    #a) calculating V(s') by querying this critic network (ie calling 'forward') with pairs of next_ob_no
                    #b) and computing the target values as r(s1, a1) - r(s2, a2) + gamma * V(s')
                # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it to 0) when a terminal state is reached
            # every time,
                # update this critic using the observations and targets
                # HINT1: need to sess.run the following: 
                    #a) critic_update_op 
                    #b) critic_loss
                # HINT2: need to populate the following (in the feed_dict): 
                    #a) sy_ob_no with ob_no
                    #b) sy_target_n with target values calculated above
        if self.sample_strat == "mixed":
            rand_samp = ob_no.shape[0]//2
        else:
            rand_samp = ob_no.shape[0]
        if self.sample_strat == "constrained_random":
            rand_first_inds = np.random.choice(ob_no.shape[0]-10, size=rand_samp)
            rand_second_inds = rand_first_inds + np.random.choice(10, rand_samp) + 1
        else:
            rand_first_inds = np.random.choice(ob_no.shape[0], size=rand_samp)
            rand_second_inds = np.random.choice(ob_no.shape[0], size=rand_samp)
        
            
        def _slice(arr):
            # Ensure that returned arrays are the same length, even if the input
            # had an odd length
            
            if self.sample_strat == "sequential":
                first_half, second_half = arr[:-1], arr[1:]
                return second_half, first_half
            if self.sample_strat == "ordered_random":
                rand_t_inds = np.minimum(rand_first_inds, rand_second_inds)
                rand_tp1_inds= np.maximum(rand_second_inds, rand_first_inds)
                rand_first = arr[rand_t_inds]
                rand_second = arr[rand_tp1_inds]
                return rand_second, rand_first
            elif self.sample_strat == "constrained_random":
                rand_first = arr[rand_first_inds]
                rand_second = arr[rand_second_inds]
                return rand_second, rand_first
            elif self.sample_strat == "pure_random":
                rand_first = arr[rand_first_inds]
                rand_second = arr[rand_second_inds]
                return rand_first, rand_second
            elif self.sample_strat == "mixed":
                half_point = arr.shape[0]//2
                first_half = arr[half_point:-1]
                second_half = arr[half_point+1:]
                rand_t_inds = np.minimum(rand_first_inds, rand_second_inds)
                rand_tp1_inds= np.maximum(rand_second_inds, rand_first_inds)
                rand_first = arr[rand_first_inds]
                rand_second = arr[rand_second_inds]
                agg_first_half = np.concatenate((rand_second, second_half), axis=0)
                agg_second_half = np.concatenate((rand_first, first_half), axis=0)
                return rand_second, rand_first


        total_grad_steps = self.num_grad_steps_per_target_update * self.num_target_updates
        ob_1, ob_2 = _slice(ob_no)
        next_ob_1, next_ob_2 = _slice(next_ob_no)
        terminal_n_1, terminal_n_2 = _slice(terminal_n)
        re_n_1, re_n_2 = _slice(re_n)

        if self.terminal_val == "learn":
            for i in range(total_grad_steps):
                if i % self.num_grad_steps_per_target_update == 0:
                    # Update regular single value function  
                    v_next = self.single_forward(next_ob_no)
                    v_next = v_next * (1 - terminal_n)
                    single_target_vals = re_n + self.gamma*v_next

                single_loss, _ = self.sess.run([self.critic_loss, self.critic_update_op], feed_dict = {self.sy_ob_no: ob_no, self.sy_target_n: single_target_vals})

        for i in range(total_grad_steps):
            if i % self.num_grad_steps_per_target_update == 0:
                diff_v_next = self.forward(next_ob_1, next_ob_2)
                if self.terminal_val == "learn":
                    v_next_ob1 = self.single_forward(next_ob_1) 
                    v_next_ob2 = self.single_forward(next_ob_2)
                #v_next_ob1 = np.clip(v_next_ob1, -25, 25)
                #else:
                # TODO deal with terminal states in a smarter way
                diff_v_next = diff_v_next * (1 - terminal_n_1) * (1 - terminal_n_2)
                if self.terminal_val == "learn":
                    mask1 = -v_next_ob2 * terminal_n_1 
                    mask2 = self.gamma * v_next_ob1 * terminal_n_2
                else:
                    mask1 = -int(self.terminal_val) * terminal_n_1 
                    mask2 = self.gamma * int(self.terminal_val) * terminal_n_2
                diff_v_next += mask1 + mask2 
                #print(np.logical_and(terminal_n_1, terminal_n_2))
                diff_v_next *= np.logical_not(np.logical_and(terminal_n_1, terminal_n_2))
                #print(np.mean(re_n_1))v_next_ob1 

                target_vals = (self.gamma*re_n_1 - re_n_2) + self.gamma*diff_v_next
                #print(target_vals[:10])
                #print(v_next[:5])
                # Update regular single value function  
            ob_feed = np.concatenate((ob_1, ob_2), axis=1)
            loss, _ = self.sess.run([self.diff_critic_loss, self.diff_critic_update_op], feed_dict = {self.diff_sy_ob_no: ob_feed, self.diff_sy_target_n: target_vals})
        return loss
