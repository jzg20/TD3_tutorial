import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import os
import imageio
from stable_baselines3.common.noise import NormalActionNoise

# Create the environment with gymnasium and InvertedPendulum-v5
env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')

# Path to save the models and videos
save_path = "./inverted_pendulum_td3"
os.makedirs(save_path, exist_ok=True)

# Callback to save video of the best episode and track rewards and actions
class VideoSavingCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(VideoSavingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_episode_reward = -np.inf
        self.best_video = None
        self.video_frames = []
        self.episode_reward = 0  # Initialize reward for episode
        self.episode_rewards = []  # Track rewards for each episode
        self.episode_lengths = []  # Track lengths of episodes
        self.all_actions = []  # Track actions for action distribution plot
        self.cumulative_rewards = []  # Track cumulative reward for learning curve
        self.current_episode_length = 0  # Initialize episode length counter
        self.time_step_counter = 0
        self.best_reward_list = []

    def _on_step(self) -> bool:
        # Accumulate the reward for each step in the episode
        self.episode_reward += self.locals['rewards'][0]
        
        # Track actions taken during training for action distribution
        action = self.locals['actions'][0]
        self.all_actions.append(action)
        
        # Track the length of the current episode
        self.current_episode_length += 1
        
        # Update the cumulative reward at each time step
        self.cumulative_rewards.append(self.episode_reward)
        
        # Check if the current episode is finished
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.episode_reward)  # Append episode reward
            self.episode_lengths.append(self.current_episode_length)  # Append episode length
            if self.episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.episode_reward
                # Save the current episode video
                self.best_video = True
                self.video_frames = []  # Reset frames for the best episode
            
            # Reset the episode reward and length for the next episode
            self.episode_reward = 0
            self.current_episode_length = 0
        
        # save best reward for given time step
        self.best_reward_list.append(self.best_episode_reward)

        # Record video frames for the best episode
        if self.best_video:
            frame = self.locals['env'].render()
            # Ensure the frame has the correct shape (height, width, channels)
            if frame is not None and frame.ndim == 3:
                self.video_frames.append(frame)
        
        # Save video after 500 time steps every 4000 time steps
        if self.time_step_counter == 500 or (self.time_step_counter % 4000 == 0 and self.time_step_counter > 0):
            self.save_video_at_interval()

        self.time_step_counter += 1
        
        # Plot action distribution and episode visualizations after 500 time steps and every 4000 time steps
        if self.time_step_counter == 500 or self.time_step_counter % 4000 == 0:
            self.plot_action_distribution()
            self.plot_smoothed_rewards()
            self.plot_best_reward()
            self.plot_learning_curve()
            self.plot_reward_vs_episode()

        return True

    def _on_training_end(self):
        # Plot reward vs episode after training ends
        self.plot_reward_vs_episode()

        # Plot the learning curve (reward vs time steps)
        self.plot_learning_curve()

    def _save_video(self, video_frames, filename):
        try:
            imageio.mimsave(filename, video_frames)
        except ValueError as e:
            print(f"Error saving video: {e}")

    def save_video_at_interval(self):
        # Save video every 4000 time steps
        video_filename = os.path.join(self.save_path, f"episode_video_{self.time_step_counter}.mp4")
        print(f"Saving video at time step {self.time_step_counter} to {video_filename}")
        if self.video_frames:
            self._save_video(self.video_frames, video_filename)
            self.video_frames = []  # Reset the frames after saving

    def plot_action_distribution(self):
        # Plot the action distribution for the all actions
        plt.figure(figsize=(8, 6))
        plt.hist(np.array(self.all_actions), bins=50, alpha=0.6, color='g', edgecolor='black')
        plt.xlabel("Action Value")
        plt.ylabel("Frequency")
        plt.title("Action Distribution")
        plt.savefig(os.path.join(self.save_path, f"action_distribution_{self.time_step_counter}.png"))
        plt.close()
        
    def plot_reward_vs_episode(self):
        # Plot reward vs episode after training ends
        plt.figure(figsize=(8, 6))
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.savefig(os.path.join(self.save_path, f"reward_vs_episode_{self.time_step_counter}.png"))
        plt.close()

    def plot_learning_curve(self):
        # Plot reward vs time steps (learning curve)
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.cumulative_rewards)), self.cumulative_rewards, label="Cumulative Reward")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Reward")
        plt.title("Learning Curve: Reward vs Time Steps")
        plt.savefig(os.path.join(self.save_path, f"learning_curve_{self.time_step_counter}.png"))
        plt.close()

    def plot_smoothed_rewards(self):
        smoothed_rewards = self._smooth(self.episode_rewards, window_size=50)
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Smoothed Reward vs Episode")
        plt.savefig(os.path.join(self.save_path, f"smoothed_reward_vs_episode_{self.time_step_counter}.png"))
        plt.close()
    
    def plot_best_reward(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.best_reward_list)), self.best_reward_list, label="Best Reward")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.title("Best Reward vs Time Step")
        plt.savefig(os.path.join(self.save_path, f"best_reward_vs_time_step_{self.time_step_counter}.png"))
        plt.close()
  
    def _smooth(self, data, window_size=50):
        if len(data) < window_size:
            return data
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode="valid")
        return smoothed
    
# Function to apply moving average (smoothing)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to run the experiment for multiple runs
def run_experiment(num_runs=5):
    all_cumulative_rewards = []
    all_actions_all_runs = []
    
    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}")
        
        # Create the environment with gymnasium and InvertedPendulum-v5
        env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')
        
        # Define action noise for exploration
        action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape))
        
        # Create the model (TD3) with action noise
        model = TD3("MlpPolicy", env, action_noise=action_noise)
        
        # Path to save the models and videos
        run_save_path = os.path.join(save_path, f"run_{run + 1}")
        os.makedirs(run_save_path, exist_ok=True)

        # Create a new callback for each run
        video_callback = VideoSavingCallback(run_save_path)

        # Train the model with 10000 time steps
        model.learn(total_timesteps=100001, callback=video_callback)

        # Collect the cumulative rewards from the callback
        all_cumulative_rewards.append(video_callback.cumulative_rewards)
        all_actions_all_runs.append(video_callback.all_actions)
                
        # Save the final model
        model.save(os.path.join(run_save_path, f"td3_inverted_pendulum_run_{run + 1}"))

    # After all runs, calculate average and standard deviation of rewards
    return np.array(all_cumulative_rewards), np.array(all_actions_all_runs)

# Run the experiment for 5 runs (modify num_runs as needed)
num_runs = 10
all_cumulative_rewards, all_actions_all_runs = run_experiment(num_runs)

# Plot all learning curves, average, and standard deviation
plt.figure(figsize=(10, 8))
for run in range(num_runs):
    plt.plot(all_cumulative_rewards[run], label=f"Run {run + 1}", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Cumulative Reward")
plt.title("Learning Curve: Reward vs Time Steps for Multiple Runs")
plt.legend()
plt.savefig(os.path.join(save_path, "learning_curve_smoothed.png"))
plt.close()

# Plot all learning curves, average, and standard deviation
plt.figure(figsize=(10, 8))
# Calculate and plot the average learning curve
avg_rewards = np.mean(all_cumulative_rewards, axis=0)
std_rewards = np.std(all_cumulative_rewards, axis=0)
plt.plot(avg_rewards, label="Average Reward", color="black", linewidth=2)
plt.fill_between(range(len(avg_rewards)), avg_rewards - std_rewards, avg_rewards + std_rewards, color="gray", alpha=0.3)
plt.xlabel("Time Steps")
plt.ylabel("Average Cumulative Reward")
plt.title("Learning Curve: Average Reward vs Time Steps for Multiple Runs")
plt.legend()
plt.savefig(os.path.join(save_path, "average_learning_curve.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(np.mean(all_actions_all_runs), bins=50, alpha=0.6, color='g', edgecolor='black')
plt.xlabel("Action Value")
plt.ylabel("Frequency")
plt.title("Action Distribution")
plt.savefig(os.path.join(save_path, f"action_distribution.png"))
plt.close()
