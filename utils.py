import numpy as np
import tensorflow as tf

def task_1_loss_fn_np(x, y):
	# return np.square(x + 0.2) + np.square(y)
	return 10*np.log(np.maximum([x + y], [0.000000005])) - 4*np.minimum([x], [100]) + 4*np.maximum([y], [100])
	# return np.array([x*x + y*y + 2*x*y])

def task_2_loss_fn_np(x, y):
	# return np.square(x) + np.square(y + 0.2)
	return 5*np.log(np.maximum([x + y + 0.3], [0.000000005])) - 4*np.minimum([x], [100]) + 4*np.maximum([y], [100])
	# return np.array([x*x + y*y + 2*x*y])

def multi_task_loss_fn_np(x, y):
	return task_1_loss_fn_np(x, y) + task_2_loss_fn_np(x, y)

def task_1_loss_fn_tf(theta_1, theta_2):
	return 10*tf.log(tf.maximum(theta_1 + theta_2, 0.000000005)) - 4*tf.minimum(theta_1, 100) + 4*tf.maximum(theta_2, 100)
	# return x*x + y*y + 2*x*y

def task_2_loss_fn_tf(theta_1, theta_2):
	return 5*tf.log(tf.maximum(theta_1 + theta_2 + 0.3, 0.000000005)) - 4*tf.minimum(theta_1, 100) + 4*tf.maximum(theta_2, 100)
	# return x*x + y*y + 2*x*y

def compute_corrected_gradient(grad_1, grad_2):
	projection_1_onto_2 = tf.tensordot(grad_1, grad_2, 1) * grad_2 / tf.tensordot(grad_2, grad_2, 1)
	conditional = projection_1_onto_2[0] / grad_2[0] > 0.
	return tf.cond(conditional, lambda: grad_1, lambda: grad_1 - projection_1_onto_2)

def make_plot(thetas_1_arr, thetas_2_arr, loss_arr, loss_function, task_id):
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	x_line = np.linspace(-INIT_RANGE, INIT_RANGE, 100)
	y_line = np.linspace(-INIT_RANGE, INIT_RANGE, 100)

	x_vals = []
	y_vals = []
	z_vals = []
	for x in x_line:
		for y in y_line:
			x_vals.append(x)
			y_vals.append(y)
			z_vals.append(loss_function(x, y)[0])

	ax.plot3D(x_vals, y_vals, z_vals, 'gray')
	ax.scatter3D(thetas_1_arr, thetas_2_arr, loss_arr, c=np.arange(len(loss_arr)), cmap='rainbow')

	plt.savefig('illustrative_example_task_{}_gradient_surgery={}.png'.format(task_id, gradient_surgery))