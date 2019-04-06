# The point of this is learning by supervised training. ie: You use the rollouts from the expert
# policies and you train your network to output the same action when you observe the same thing

import tensorflow as tf
tf.enable_eager_execution()
import gym
import numpy as np
import os
from tqdm import tqdm
import pickle
from absl import flags, app

flags.DEFINE_string('envname', 'Ant-v2', 'Name of the env')
flags.DEFINE_integer('steps', 10000,  'Number of training steps')
flags.DEFINE_boolean('render', False, 'Render the env')
flags.DEFINE_float('lr', 1e-4, 'Learning rate')
flags.DEFINE_integer('num_rollouts', 10, 'Number of evalution rollouts')
flags.DEFINE_list('hidden', '100,100', 'network hidden size')
FLAGS = flags.FLAGS


def build_model(hidden, input_shape, number_outputs):
    m = []
    for i, h in enumerate(hidden):
        if i == 0:
            m.append(tf.keras.layers.Dense(h, activation=tf.nn.relu, input_shape=input_shape))
        else:
            m.append(tf.keras.layers.Dense(h, activation=tf.nn.relu))
        m.append(tf.keras.layers.Dropout(0.3))
    m.append(tf.keras.layers.Dense(number_outputs))
    return tf.keras.Sequential(m)


def main(args):
    del args
    # 1) Load the Gym env and introspect the action and observation space
    env = gym.make(FLAGS.envname)
    hidden = [int(x) for x in FLAGS.hidden]
    print("Model: {}".format(hidden))
    model = build_model(
        hidden, env.observation_space.shape, env.action_space.shape[0])
    model.summary()
    # 2) Load the expert rollouts
    with open(os.path.join('expert_data', FLAGS.envname + '.pkl'), 'rb') as f:
        rollouts = pickle.loads(f.read())
    observations, actions = rollouts['observations'], rollouts['actions']
    observations = observations.astype(np.float32)
    actions = actions.astype(np.float32)
    assert observations.shape[0] == actions.shape[0]
    print(observations.shape, actions.shape)
    if len(actions.shape) == 3:
        new_actions = np.empty([actions.shape[0], actions.shape[2]])
        new_actions = actions[:,0,:]
        actions = new_actions
    print("Training with {} rollouts".format(observations.shape[0]))
    # 3) Train on the rollouts
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions))
    dataset = dataset.repeat().batch(64).shuffle(observations.shape[0])
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    for (_, (obs, actions)) in enumerate(tqdm(dataset.take(FLAGS.steps), total=FLAGS.steps)):
        with tf.GradientTape() as tape:
            predicted_action = model(obs, training=True)
            error = tf.losses.mean_squared_error(predicted_action, actions)

        grads = tape.gradient(error, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables),
                              global_step=tf.train.get_or_create_global_step())
    # 4) Save the model
    output_dir = os.path.join("models", FLAGS.envname, str(hidden))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'model.h5'))
    # 5) Evaluate the model
    returns = []
    for i in range(FLAGS.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = obs.astype(np.float32)
            action = model(obs[None, :])
            obs, r, done, _ = env.step(action[np.newaxis, :])
            totalr += r
            steps += 1
            if FLAGS.render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, env.spec.timestep_limit))
            if steps >= env.spec.timestep_limit:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(returns, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
