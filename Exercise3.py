import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') # use ggplot style
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Python version ' + sys.version)
print('Tensorflow version ' + tf.__version__)
print('Pandas version ' + pd.__version__)
print('Numpy version ' + np.__version__)

# we are going to model the following formula: y = m * x + b

# let's generate 100 random samples
train_x = np.random.rand(100).astype(np.float32)

# let's compute train_y using 0.1. for m and 0.3 for b
train_y = 0.1 * train_x + 0.3

df = pd.DataFrame({'x':train_x,
				   'y':train_y})
				   
print(df.head())
print(df.describe())

df.plot.scatter(x='x', y='y', figsize=(15,5))

test_x = np.random.rand(100).astype(np.float32)

# placeholders
x = tf.placeholder(tf.float32, name="01_x")
y = tf.placeholder(tf.float32, name="01_y")

# variables
W = tf.Variable(np.random.rand())
b = tf.Variable(np.random.rand())
pred = W * x + b

# minimize the mean squared errors
loss = tf.reduce_mean(tf.square(pred - y))

# we pick our optimizer and learning rate
optimizer = tf.train.GradientDescentOptimizer(0.7)

# we train our model by minimizing our loss function
train = optimizer.minimize(loss)

# initialize the variables
init = tf.global_variables_initializer()

# run your graph
with tf.Session() as sess:
	sess.run(init)
	
	# fit the function
	for step in range(200):
		# get your data
		train_data = {x:train_x, y:train_y}
		
		# training in progress
		sess.run(train, feed_dict=train_data)
		
		# print the last 20 results
		if step > 180:
			print(step, sess.run(W), sess.run(b))
			
	# note that W and b match the line we are trying to model (y = 0.1x + 0.3)
	print('Training completed: ', 'W=', sess.run(W), 'b=', sess.run(b))
	
	# run your train model on the test data set
	test_results = sess.run(pred, feed_dict={x:test_x})
	
	# capture the predcited results so you can plot them
	df_final = pd.DataFrame({'test_x':test_x,
							 'pred':test_results})
	
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
	
	# Chart 1 - Shows the line we are trying to model
	df.plot.scatter(x='x', y='y', ax=axes, color='red')
	
	# Chart 2 - Shows the line our trained model come up with
	df_final.plot.scatter(x='test_x', y='pred', ax=axes, alpha=0.3)
	
	# add a little sugar
	axes.set_title('target vs pred', fontsize=20)
	axes.set_ylabel('y', fontsize=15)
	axes.set_xlabel('x', fontsize=15)	
		
	plt.show()

