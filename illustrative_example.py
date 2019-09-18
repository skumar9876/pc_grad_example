import numpy as np
import tensorflow as tf
from utils import *

INIT_RANGE = 10
gradient_surgery = False




###############
# Build graph.
###############

# Fixing random seeds for initializing theta_1 and theta_2 for reproducibility.
random_uniform_initializer_1 = tf.random_uniform_initializer(-INIT_RANGE, INIT_RANGE, seed=1)
random_uniform_initializer_2 = tf.random_uniform_initializer(-INIT_RANGE, INIT_RANGE, seed=2)
theta_1 = tf.get_variable("theta_1", [], initializer=random_uniform_initializer_1, dtype=tf.float32)
theta_2 = tf.get_variable("theta_2", [], initializer=random_uniform_initializer_2, dtype=tf.float32)

loss_1 = task_1_loss_fn_tf(theta_1, theta_2)
loss_2 = task_2_loss_fn_tf(theta_1, theta_2)

loss = loss_1 + loss_2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

if not gradient_surgery:
	train_op = optimizer.minimize(loss)
else:
	# Compute corrected gradients.
	task_1_grad_theta_1 = optimizer.compute_gradients(loss_1, var_list=[theta_1])
	task_1_grad_theta_2 = optimizer.compute_gradients(loss_1, var_list=[theta_2])
	task_1_grads = tf.stack([task_1_grad_theta_1[0][0], task_1_grad_theta_2[0][0]], axis=0)

	task_2_grad_theta_1 = optimizer.compute_gradients(loss_2, var_list=[theta_1])
	task_2_grad_theta_2 = optimizer.compute_gradients(loss_2, var_list=[theta_2])
	task_2_grads = tf.stack([task_2_grad_theta_1[0][0], task_2_grad_theta_2[0][0]], axis=0)

	corrected_task_1_grads = compute_corrected_gradient(task_1_grads, task_2_grads)
	corrected_task_2_grads = compute_corrected_gradient(task_2_grads, task_1_grads)

	# Collect corrected grads and vars.
	grads_and_vars = []
	thetas = [theta_1, theta_2]
	for i in range(len(thetas)):
		grads_and_vars.append((corrected_task_1_grads[i], thetas[i]))
		grads_and_vars.append((corrected_task_2_grads[i], thetas[i])) 

	# Apply gradients.
	train_op = optimizer.apply_gradients(grads_and_vars)

###############
# Run graph.
###############

sess = tf.Session()
sess.run(tf.global_variables_initializer())

thetas_1_arr = []
thetas_2_arr = []
loss_1_arr = []
loss_2_arr = []
loss_arr = []
for i in range(100):
	loss_1_, loss_2_, loss_, theta_1_, theta_2_ = sess.run(
		[loss_1, loss_2, loss, theta_1, theta_2])
	_ = sess.run(train_op)
	
	loss_1_arr.append(loss_1_)
	loss_2_arr.append(loss_2_)
	loss_arr.append(loss_)
	thetas_1_arr.append(theta_1_)
	thetas_2_arr.append(theta_2_)

thetas_1_arr = np.squeeze(thetas_1_arr)
thetas_2_arr = np.squeeze(thetas_2_arr)

###############
# Plot results.
###############
make_plot(thetas_1_arr, thetas_2_arr, loss_1_arr, task_1_loss_fn_np, 1, INIT_RANGE, gradient_surgery)
make_plot(thetas_1_arr, thetas_2_arr, loss_2_arr, task_2_loss_fn_np, 2, INIT_RANGE, gradient_surgery)
make_plot(thetas_1_arr, thetas_2_arr, loss_arr, multi_task_loss_fn_np, 3, INIT_RANGE, gradient_surgery)