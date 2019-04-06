# The point of this is learning by supervised training. ie: You use the rollouts from the expert
# policies and you train your network to output the same action when you observe the same thing

import tensorflow as tf
import gym
import numpy as np
import os
from tqdm import tqdm
import pickle
from absl import flags, app
import load_policy_eager

flags.DEFINE_string('envname', 'Ant-v2', 'Name of the env')

flags.DEFINE_integer('steps', 4000,  'Number of training steps on the rollout dataset')
flags.DEFINE_integer('generation_rollouts', 5,  'Number of rollouts you generate to then be annotated by the expert policy')
flags.DEFINE_integer('iterations', 5, 'Number of iterations of the algorithm')

flags.DEFINE_boolean('render', False, 'Render the env')
flags.DEFINE_float('lr', 1e-4, 'Learning rate')
flags.DEFINE_integer('evaluation_rollouts', 10, 'Number of evalution rollouts')
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

def generate_rollouts(env, model, num_rollouts, model_action_to_env_action=lambda x: x, render=False):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('Generating rollout number {}'.format(i + 1))
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = obs.astype(np.float32)
            action = model(obs[None, :])
            action = model_action_to_env_action(action)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            if render:
                env.render()
            totalr += r
            steps += 1
            if steps >= env.spec.timestep_limit:
                break
        returns.append(totalr)
    return np.array(observations), np.array(actions), returns


def dagger(env, dataset=None, optim=None, model=None, training_steps=None, generation_rollouts=None, expert_policy_fn=None):
    #Run one iteration of dagger and return a new array of {observations, actions} annotated by the expert policy
    #1) Train the model on the dataset for training_steps step
    for (_, (obs, actions)) in enumerate(tqdm(dataset.take(training_steps), total=training_steps)):
        with tf.GradientTape() as tape:
            predicted_action = model(obs, training=True)
            error = tf.losses.mean_squared_error(predicted_action, actions)

        grads = tape.gradient(error, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables),
                              global_step=tf.train.get_or_create_global_step())
    #2) Generate rollouts from the model
    observations, _, _ = generate_rollouts(env, model, generation_rollouts, model_action_to_env_action=lambda action: action[np.newaxis, :])
    #3) Annotate the observations by the expert policy
    print("Annotation...")
    annotated_actions = expert_policy_fn(observations)[:, None, :]
    print("Annotation done")
    return observations, annotated_actions


def generate_dataset(observations, actions):
    observations = observations.astype(np.float32)
    actions = actions.astype(np.float32)
    assert observations.shape[0] == actions.shape[0]
    if len(actions.shape) == 3:
        new_actions = np.empty([actions.shape[0], actions.shape[2]])
        new_actions = actions[:,0,:]
        actions = new_actions
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions))
    dataset = dataset.repeat().batch(64).shuffle(observations.shape[0])
    return dataset



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
    # 3) Load the expert policy
    policy_fn = load_policy_eager.load_policy(os.path.join("experts", FLAGS.envname + ".pkl"))
    # Set up everything
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    # Run dagger iterations time
    for iteration in range(FLAGS.iterations):
        print('Dagger iteration {}'.format(iteration + 1))
        dataset = generate_dataset(observations, actions)
        obs, acts = dagger(env, dataset=dataset, optim=optim, model=model, training_steps=FLAGS.steps, generation_rollouts=FLAGS.generation_rollouts, expert_policy_fn=policy_fn)
        observations = np.vstack((observations,obs))
        actions = np.vstack((actions , acts))

    # 4) Save the model
    output_dir = os.path.join("dagger_models", FLAGS.envname, str(hidden))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'model.h5'))
    # 5) Evaluate the model
    _, _, returns  = generate_rollouts(env, model, FLAGS.evaluation_rollouts, model_action_to_env_action=lambda action: action[np.newaxis, :], render=FLAGS.render)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(returns, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
