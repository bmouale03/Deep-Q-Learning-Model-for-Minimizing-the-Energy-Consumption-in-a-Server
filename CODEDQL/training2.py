# Artificial Intelligence for Business - Case Study 2
# Training the AI

# Importing the libraries and the other python files
import os

# 🔹 Empêcher TensorFlow d'afficher des logs inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random as rn
import environment
import brain_nodropout
import dqn
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
# 🔹 Supprimer les logs verbeux de TensorFlow/Keras
tf.get_logger().setLevel('ERROR')

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# BUILDING THE ENVIRONMENT
env = environment.Environment(optimal_temperature=(18.0, 24.0),
                              initial_month=0,
                              initial_number_users=20,
                              initial_rate_data=30)

# BUILDING THE BRAIN
brain = brain_nodropout.Brain(learning_rate=0.00001,
                              number_actions=number_actions)

# BUILDING THE DQN MODEL
dqn = dqn.DQN(max_memory=max_memory, discount=0.9)

# CHOOSING THE MODE
train = True

# TRAINING THE AI
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

# 🔹 Stockage des métriques
rewards = []
losses = []

# 🔹 Préparer le graphique
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].set_title("Total Reward per Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Reward")

ax[1].set_title("Average Loss per Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")

if env.train:
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        steps = 0
        new_month = np.random.randint(0, 12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0

        while (not game_over) and timestep <= 5 * 30 * 24 * 60:
            # Exploration vs exploitation
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                direction = -1 if (action - direction_boundary < 0) else 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            else:
                q_values = model.predict(current_state, verbose=0)  # silence
                action = np.argmax(q_values[0])
                direction = -1 if (action - direction_boundary < 0) else 1
                energy_ai = abs(action - direction_boundary) * temperature_step

            # Update environment
            next_state, reward, game_over = env.update_env(
                direction, energy_ai, int(timestep / (30 * 24 * 60))
            )
            total_reward += reward

            # Store experience
            dqn.remember([current_state, action, reward, next_state], game_over)

            # Training on batch (🔹 avec sécurité)
            inputs, targets = dqn.get_batch(model, batch_size=batch_size)
            if inputs is not None and targets is not None:
                loss += model.train_on_batch(inputs, targets, return_dict=False)
                steps += 1
            else:
                # ⚠️ Mémoire pas encore suffisante → on attend
                pass

            timestep += 1
            current_state = next_state

        avg_loss = loss / steps if steps > 0 else 0

        rewards.append(total_reward)
        losses.append(avg_loss)

        # Résumé epoch
        print("\n")
        print(f"Epoch: {epoch:03d}/{number_epochs}")
        print(f"Total Energy spent with an AI: {env.total_energy_ai:.0f}")
        print(f"Total Energy spent with no AI: {env.total_energy_noai:.0f}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Loss: {avg_loss:.6f}")

        # Mise à jour du graphique
        ax[0].plot(rewards, color="blue")
        ax[1].plot(losses, color="red")
        plt.pause(0.01)

        # Early stopping
        if early_stopping:
            if total_reward <= best_total_reward:
                patience_count += 1
            else:
                best_total_reward = total_reward
                patience_count = 0
            if patience_count >= patience:
                print("Early Stopping")
                break

        # Saving model
        model.save("model.h5")

# 🔹 Sauvegarde finale du graphique
plt.ioff()
plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
