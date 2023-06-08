import gymnasium as gym
import numpy as np
from stable_baselines3.dqn import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          df = load_results(self.log_dir)
          x, y = ts2xy(df, 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folders, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    for log_folder,tag in log_folders:
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y,label=f"{tag}")
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.legend(loc='upper left', fontsize='small')
    plt.title(title + " Smoothed")
    plt.show()


if __name__ == "__main__":
    log_dir = "/tmp/gym/DQN/CartPole"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("CartPole-v1",  # name of your gym environement
                   render_mode=None,  # "human"
                   )
    env = Monitor(env, log_dir,info_keywords=("learning_rate","gamma"))

    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
    model = DQN("MlpPolicy",  # type de réseaux de neuronnes
                env,  # passe l'environement
                verbose=1,  # mode verbeux ( affiche plus d'information )
                learning_rate=0.0001,  # un paramètre que tu devras tester
                gamma=0.99,  # un paramètre que tu devras tester
                )
    trained_model = model.learn(
        total_timesteps=100000,  # temps d'entrainement
        log_interval=4,  # intervalle de remonter des données
        callback=callback,  # Utiliser pour stocker des métriques

    )
    log_dir = "/tmp/gym/DQN/CartPole"
    plot_results(log_dir)
