#Hidden Markov Model
#we get observations that give us info for predictions on our model

# We are using a different module from tensorflow this time
import tensorflow as tf
import tensorflow_probability as tfp



# Weather Model
# Taken direclty from the TensorFlow documentation (https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel).
#
# We will model a simple weather system and try to predict the temperature on each day given the following information.
#
# Cold days are encoded by a 0 and hot days are encoded by a 1.
# The first day in our sequence has an 80% chance of being cold.
# A cold day has a 30% chance of being followed by a hot day.
# A hot day has a 20% chance of being followed by a cold day.
# On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
# If you're unfamiliar with standard deviation it can be put simply as the range of expected values.
#
# In this example, on a hot day the average temperature is 15 and ranges from 5 to 25.
#
# To model this in TensorFlow we will do the following.

# the loc argument represents the mean and the scale is the standard devitation
tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.3, 0.8])
transition_distribution = tfd.Categorical(probs=[[0.2, 0.5],
                                                 [0.2, 0.8]])

# the loc argument represents the mean and the scale is the standard devitation
observation_distribution = tfd.Normal(loc=[0., 100.], scale=[20., 80.])

#We've now created distribution variables to model our system and it's time to create the hidden markov model.
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)


# The number of steps represents the number of days that we would like to predict information for.
# In this case we've chosen 7, an entire week.
# To get the expected temperatures on each day we can do the following.
mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor
# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
    print('Finally Probabilitys for day Temperatures')
    print(mean.numpy())

