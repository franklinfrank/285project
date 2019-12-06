from .bootstrapped_continuous_critic import BootstrappedContinuousCritic
import tensorflow as tf
from cs285.infrastructure.tf_utils import build_mlp
import numpy as np

class DifferentialCritic(BootstrappedContinuousCritic):
    def __init__(self, sess, hparams):
        super().__init__(sess, hparams)
    
    def _build(self):

        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # define the critic
        self.critic_prediction = tf.squeeze(build_mlp(
            self.sy_ob_no,
            1,
            "nn_critic",
            n_layers=self.n_layers,
            size=self.size))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)

        # TODO: set up the critic loss
        # HINT1: the critic_prediction should regress onto the targets placeholder (sy_target_n)
        # HINT2: use tf.losses.mean_squared_error
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
        sy_ob_no = tf.placeholder(shape=[None, 2*self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n
    
    def forward(self, ob_1, ob_2):
            # TODO: run your critic
            # HINT: there's a neural network structure defined above with mlp layers, which serves as your 'critic'
            ob = np.concatenate((ob_1, ob_2), axis=1)
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
                    
        def _slice(arr):
            # Ensure that returned arrays are the same length, even if the input
            # had an odd length
            slice_ind = arr.shape[0]//2
            first_half, second_half = arr[:2*slice_ind:2], arr[1:2*slice_ind:2]
            first_half_trunc = arr[::10]
            agg_first_half = np.concatenate((first_half, second_half), axis=0)
            agg_second_half = np.concatenate((second_half, first_half), axis=0)
            return agg_first_half, agg_second_half
            #return first_half, second_half

        total_grad_steps = self.num_grad_steps_per_target_update * self.num_target_updates
        ob_1, ob_2 = _slice(ob_no)
        next_ob_1, next_ob_2 = _slice(next_ob_no)
        terminal_n_1, terminal_n_2 = _slice(terminal_n)
        re_n_1, re_n_2 = _slice(re_n)
     
        for i in range(total_grad_steps):
            if i % self.num_grad_steps_per_target_update == 0:
                v_next = self.forward(next_ob_1, next_ob_2)
                if self.gamma != 1:
                    v_next_ob1 = self.forward(next_ob_1, next_ob_1) / (self.gamma - 1)
                    v_next_ob2 = self.forward(next_ob_2, next_ob_2) / (self.gamma - 1)
                else:
                    v_next_ob1, v_next_ob2 = 0, 0
                # TODO deal with terminal states in a smarter way
                v_next = v_next * (1 - terminal_n_1) * (1-terminal_n_2) 
                mask1 = -v_next_ob2 * terminal_n_1 
                mask2 = self.gamma * v_next_ob1 * terminal_n_2
                
                v_next += mask1 + mask2 
                v_next *= np.logical_not(np.logical_and(terminal_n_1, terminal_n_2))

                target_vals = np.clip(self.gamma*re_n_1 - re_n_2 + self.gamma*v_next, -50, 50)
            ob_feed = np.concatenate((ob_1, ob_2), axis=1)
            loss, _ = self.sess.run([self.critic_loss, self.critic_update_op], feed_dict = {self.sy_ob_no: ob_feed, self.sy_target_n: target_vals})
        return loss
