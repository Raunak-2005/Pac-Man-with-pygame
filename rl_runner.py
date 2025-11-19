# rl_runner.py - FINAL VERSION WITH GRAPHICS CHOICE
from game import Game
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import pygame
import os
import time
import signal
import sys

# === Metrics ===
episode_rewards = []
episode_times = []
batch_losses = []
epsilons = []
victory_episodes = []

agent = DQNAgent(state_size=16, action_size=4)
os.makedirs("plots", exist_ok=True)

# Ask user for graphics
print("PAC-MAN DEEP REINFORCEMENT LEARNING")
print("Do you want to see graphics during training?")
choice = input("Type 'yes' for full visuals every episode, 'no' for max speed: ").strip().lower()

show_graphics = choice in ['yes', 'y', '1', 'true']

if show_graphics:
    print("\nGraphics ENABLED — Full Pac-Man visuals every episode (slower)")
else:
    print("\nGraphics DISABLED — Maximum training speed (recommended for final run)")

print("\nStarting training... (Press Ctrl+C to stop early)\n")

def save_final_plots():
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    eps = range(1, len(episode_rewards)+1)

    # Reward
    plt.subplot(2, 2, 1)
    plt.plot(eps, episode_rewards, color='skyblue', linewidth=1.8, label='Total Reward')
    if len(episode_rewards) >= 100:
        ma = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(eps[99:], ma, 'red', linewidth=3, label='100-Episode Avg')
    if victory_episodes:
        plt.scatter(victory_episodes,
                [episode_rewards[i-1] for i in victory_episodes],
                olor='lime', s=300, marker='*', edgecolor='darkgreen', linewidth=2.5,
                label='VICTORY!', zorder=10)
    plt.axhline(600, color='green', linestyle='--', alpha=0.8)
    plt.title("Total Reward per Episode", fontsize=14, fontweight='bold')
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    # Loss
    plt.subplot(2, 2, 2)
    if batch_losses:
        smooth = np.convolve(batch_losses, np.ones(200)/200, mode='valid')
        plt.plot(smooth, 'purple', linewidth=2.5)
    plt.title("Training Loss (200-batch MA)")
    plt.grid(alpha=0.3)

    # Time per Episode
    plt.subplot(2, 2, 3)
    plt.plot(eps, episode_times, color='orange', linewidth=1.8)
    plt.title("Time per Episode (seconds)")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.grid(alpha=0.3)

    # Epsilon
    plt.subplot(2, 2, 4)
    plt.plot(eps, epsilons, color='gray', linewidth=2.5)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.grid(alpha=0.3)

    plt.suptitle("Pac-Man Double DQN - Training Results", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/pacman_results.png", dpi=300, bbox_inches='tight')
    plt.savefig("plots/pacman_results.pdf", bbox_inches='tight')
    plt.show()

def signal_handler(sig, frame):
    print(f"\nTraining stopped after {len(episode_rewards)} episodes")
    pygame.quit()
    save_final_plots()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

episode = 0
try:
    while episode < 500:
        episode += 1
        start_time = time.time()

        game = Game(agent=agent)
        state = game.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = game.step(action)

            # Capture loss
            if hasattr(agent, 'last_loss') and agent.last_loss is not None:
                batch_losses.append(agent.last_loss)

            total_reward += reward
            state = next_state

            # === GRAPHICS CONTROL ===
            if show_graphics:
                game.screen.fill((0, 0, 0))
                game.map.draw(game.screen)
                game.all_sprites.draw(game.screen)
                pygame.display.flip()
                game.clock.tick(120)
            else:
                pygame.event.pump()  # Prevent freezing even when hidden

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

        # === Episode End ===
        ep_time = time.time() - start_time
        episode_rewards.append(total_reward)
        episode_times.append(ep_time)
        epsilons.append(agent.epsilon)

        status = "VICTORY!" if game.victory else "Death"
        print(f"Ep {episode:3d} | R:{total_reward:7.1f} | T:{ep_time:5.1f}s | {status}", end="")
        if game.victory:
            victory_episodes.append(episode)
            print(" ← LEVEL CLEARED!")
        else:
            print()

        if episode == 500 and not game.victory:
            print("   → Episode 500 died → running one more until death...")

except KeyboardInterrupt:
    pass
finally:
    pygame.quit()
    save_final_plots()
    print(f"\nTraining finished! Total episodes: {len(episode_rewards)}")
    print(f"Victories: {len(victory_episodes)}")
    print("Graphs saved → plots/pacman_results.png")