from .base_critic import BaseCritic
import tensorflow as tf
from cs285.infrastructure.dqn_utils import minimize_and_clip, huber_loss
from cs285.infrastructure.tf_utils import build_mlp
import numpy as np

class DiffDQNCritic(BaseCritic):

    def __init__(self, sess, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.sess = sess
        self.env_name = hparams['env_name']
        self.act_space = hparams['n_act_dim']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        self.define_placeholders()
        self._build(hparams['value_func'], hparams['rew_func'])

    def _build(self, value_func, rew_func):

        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")

        self.flat_obs_t = tf.concat([self.obs_t_ph_1, self.obs_t_ph_2], 0) 
        self.flat_obs_tp1 = tf.concat([self.obs_tp1_ph_1, self.obs_tp1_ph_2], 0)
        self.rew_model = rew_func(tf.concat([self.flat_obs_t, tf.one_hot(self.act_t_ph, depth=self.act_space, dtype=tf.float32)], 1),
                         self.ac_dim, scope='rew_func', reuse=False)

        self.rew_full = tf.concat([self.rew_t_ph_1, self.rew_t_ph_2], 0)/100.
        self.rew_error = tf.reduce_mean((self.rew_model - self.rew_full)**2) 
        self.rew_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rew_error)
    
        #####################
        
        self.dyn_model = build_mlp(self.flat_obs_t, self.ob_dim, 'dyn_model', 3, 64)
        self.dyn_error = tf.reduce_mean((self.flat_obs_tp1 - self.dyn_model)**2)
        self.dyn_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dyn_error)

        #####################

        # q values, created with the placeholder that holds CURRENT obs (i.e., t)
        self.combined_obs_t = tf.concat([self.obs_t_ph_1, self.obs_t_ph_2], 1)
        self.combined_obs_tp1 = tf.concat([self.obs_tp1_ph_1, self.obs_tp1_ph_2], 1)
        self.values = value_func(self.combined_obs_t, self.ac_dim, scope='value_func', reuse=False)
        #####################

        # next step values, created with the placeholder that holds NEXT obs (i.e., t+1)
        v_tp1_values = value_func(self.combined_obs_tp1, self.ac_dim, scope='target_value_func', reuse=False)

        #####################

        # TODO calculate the targets for the Bellman error
        # HINT1: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (1 - self.done_mask_ph)
        # HINT2: see above, where q_tp1 is defined as the q values of the next timestep
        # HINT3: see the defined placeholders and look for the one that holds current rewards
        rew_diff = self.rew_t_ph_1 - self.rew_t_ph_2
        target_value_t = rew_diff+self.gamma*v_tp1_values*(1-self.done_mask_ph_1)*(1-self.done_mask_ph_2)
        target_value_t += self.gamma * self.done_mask_ph_1 * -10
        target_value_t += self.done_mask_ph_2 * self.gamma * 10

        target_value_t = tf.stop_gradient(target_value_t)

        #####################

        # TODO compute the Bellman error (i.e. TD error between q_t and target_q_t)
        # Note that this scalar-valued tensor later gets passed into the optimizer, to be minimized
        # HINT: use reduce mean of huber_loss (from infrastructure/dqn_utils.py) instead of squared error
        self.total_error= tf.reduce_mean(huber_loss(self.values-target_value_t))

        #####################

        # TODO these variables should all of the 
        # variables of the Q-function network and target network, respectively
        # HINT1: see the "scope" under which the variables were constructed in the lines at the top of this function
        # HINT2: use tf.get_collection to look for all variables under a certain scope
        value_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="value_func")
        target_value_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_value_func")

        #####################

        # train_fn will be called in order to train the critic (by minimizing the TD error)
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                                          var_list=value_func_vars, clip_val=self.grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(value_func_vars,        key=lambda v: v.name),
                                   sorted(target_value_func_vars , key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

    def get_best_action(self, sess, observation):
        actions = np.arange(self.act_space) 
        state_preds = []

        for act in np.arange(self.act_space):
            state_pred = sess.run(self.dyn_model, 
                                  feed_dict= {
                                      self.obs_t_ph_1: observation[::2],
                                      self.obs_t_ph_2: observation[1::2],                                                         
                                      self.act_t_ph: np.array([act] * observation.shape[0])
                                  })
            state_preds.append(state_pred)
        state_preds = np.array(state_preds)

        rew_preds = []
        for act in np.arange(self.act_space):
            rew_pred = sess.run(self.rew_model, 
                                feed_dict={
                                    self.obs_t_ph_1: observation[::2],
                                    self.obs_t_ph_2: observation[1::2],
                                    self.act_t_ph: np.array([act] * observation.shape[0])
                                })

            rew_preds.append(rew_pred)
        rew_preds = np.array(rew_preds) * 100.
        #print(rew_preds)
        big_mat = []
        for act in np.arange(self.act_space):
            per_act_q_diff = np.zeros((observation.shape[0], 1)) + 1e6 
            for act2 in np.arange(self.act_space):
                if act2 == act: 
                    continue
                value_diffs = sess.run(self.values, feed_dict=
                    {self.obs_t_ph_1: state_preds[act],
                     self.obs_t_ph_2: state_preds[act2]})
                q_diff = rew_preds[act] - rew_preds[act2] + value_diffs
                per_act_q_diff = np.minimum(np.array(q_diff), per_act_q_diff)
            big_mat.append(per_act_q_diff)
        big_mat = np.array(big_mat).squeeze(axis=2)
        #print(big_mat)
        return np.argmax(big_mat, axis=0)
               

    def define_placeholders(self):
        # set up placeholders
        # placeholder for current observation (or state)
        lander = self.env_name == 'LunarLander-v2'

        self.obs_t_ph_1 = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(self.input_shape))
        self.obs_t_ph_2 = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(self.input_shape))

        # placeholder for current action
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        # placeholder for current reward
        self.rew_t_ph_1 = tf.placeholder(tf.float32, [None])
        self.rew_t_ph_2 = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph_1 = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(self.input_shape))
        self.obs_tp1_ph_2 = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(self.input_shape))

        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph_1 = tf.placeholder(tf.float32, [None])
        self.done_mask_ph_2 = tf.placeholder(tf.float32, [None])

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        raise NotImplementedError
