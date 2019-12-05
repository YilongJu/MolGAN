import tensorflow as tf
from utils.utils import *
from keras.optimizers import Adam

class GraphGANOptimizer(object):

    def __init__(self, model, learning_rate=1e-3, feature_matching=True):
        self.la = tf.placeholder_with_default(1., shape=())

        with tf.name_scope('losses'):
            eps = tf.random_uniform(tf.shape(model.logits_real)[:1], dtype=model.logits_real.dtype)

            x_int0 = model.adjacency_tensor * tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1) + model.edges_softmax * (1 - tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1))
            x_int1 = model.node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + model.nodes_softmax * (1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))

            grad0, grad1 = tf.gradients(model.D_x((x_int0, None, x_int1), model.discriminator_units), (x_int0, x_int1))

            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + tf.reduce_mean(
                ((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keep_dims=True)

            self.loss_D = - model.logits_real + model.logits_fake
            self.loss_G = - model.logits_fake
            self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + (
                    model.value_logits_fake - model.rewardF) ** 2
            self.loss_RL = - model.value_logits_fake
            self.loss_F = (tf.reduce_mean(model.features_real, 0) - tf.reduce_mean(model.features_fake, 0)) ** 2

        self.loss_D = tf.reduce_mean(self.loss_D)
        self.loss_G = tf.reduce_sum(self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)
        self.loss_V = tf.reduce_mean(self.loss_V)
        self.loss_RL = tf.reduce_mean(self.loss_RL)
        alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))
        self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        with tf.name_scope('train_step'):
            self.train_ops_D = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.train_step_D = self.train_ops_D.minimize( loss=self.loss_D + 10 * self.grad_penalty, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

            print(f"model.unrolling_steps: {model.unrolling_steps}")

            self.train_step_z = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss_D + 10 * self.grad_penalty, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="input_LO"))

            # Unroll optimization of the discrimiantor
            if model.unrolling_steps > 1:
                grads_and_vars = self.train_ops_D.compute_gradients(self.loss_D + 10 * self.grad_penalty, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

                #print(f"grads_and_vars: {grads_and_vars}")

                # updates = self.train_ops_D.apply_gradients_fuck(grads_and_vars)
                updates = self.train_ops_D.apply_gradients(grads_and_vars)

                self.grads_and_vars = grads_and_vars
                self.printupdates = updates
                print("hahahahahaha",updates)
                # Get dictionary mapping from variables to their update value after one optimization step
                update_dict = extract_update_dict(updates)
                cur_update_dict = update_dict
                for i in range(model.unrolling_steps - 1):
                    print(f"\nunrolling - {i}\n")
                    # Compute variable updates given the previous iteration's updated variable
                    cur_update_dict = graph_replace(update_dict, cur_update_dict)
                # Final unrolled loss uses the parameters at the last time step
                self.unrolled_loss = graph_replace(self.loss_D, cur_update_dict)
                # self.unrolled_loss = -self.loss_G
            else:
                self.unrolled_loss = -self.loss_G

            # self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            #     loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) + tf.cond(
            #         tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
            #     var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
            self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=tf.cond(tf.greater(self.la, 0), lambda: -self.la * self.unrolled_loss, lambda: 0.) + tf.cond(
                    tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

            self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_V,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))
