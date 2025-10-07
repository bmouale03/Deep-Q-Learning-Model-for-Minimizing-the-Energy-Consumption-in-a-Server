#Mise en œuvre d’un algorithme de Deep Q-Learning intégrant
#le modèle de régression linéaire multiple pour l’optimisation 2nergétique des Data Center
# Training the AI
# Installing Keras
 # conda install -c conda-forge keras
# Importing the libraries and the other python files

import os # interaction avec le systeme d'exploitation
import numpy as np
import random as rn
import environnment
import brain
import dqn

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)# avoir les memes resultats lorsqu'on genere les nombres aleatoires
# SETTING THE PARAMETERS # LES PARAMETRES GLOBEAUX
epsilon = 0.3 # le parametre d'exploration (30%). Quand on entraine une IA, il fo tjrs trouver le juste milieu entre l'exploration et l'exploitation
number_actions = 5
direction_boundary = (number_actions - 1) / 2 # la frontiere de direction(pour trouver le juste milieu)
number_epochs = 100 # nombre d'epoque ( le nombre de fois qu'on fera lentrainement sur un nouveau serveur)
max_memory = 3000
batch_size = 512
temperature_step = 1.5
#BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS# CREATION DE L'ENVIRONNEMENT
env = environnment.Environnment(optimal_temperature=(18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)
# BUILDING THE BRAIN BY SIMPLY CREATING AN OBJECT OF THE BRAIN CLASS# CREATION DU CERVEAU
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)
# BUILDING THE DQN MODEL BY SIMPLY CREATING AN OBJECT OF THE DQN CLASS# CONSTRUCTION DU DEEP Q-LEARNING
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)
# CHOOSING THE MODE# CHOISIR SI ON EST EN PHASE D'ENTRAINEMENT OU DE TEST
train = True

# TRAINING THE AI# ENTRAINEMENT DE L'IA
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0
if (env.train):
    # STARTING THE LOOP OVER ALL THE EPOCHS (1 Epoch = 5 Months)
    for epoch in range(1, number_epochs):
    #INITIALIAZING ALL THE VARIABLES OF BOTH THE ENVIRONMENT AND THE TRAINING LOOP
        total_reward = 0
        loss = 0. # la fonction cout
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)#remettre a zero lenvironnement
        game_over = False
        current_state, _, _ = env.observe()# retourner 3 objets(letat initial, 0,0)
        timestep = 0
        # STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
        while ( timestep <= 5 * 30 * 24 * 60 and (not game_over)):
            #tant que le timestep est plut petit 
            # PLAYING THE NEXT ACTION BY EXPLORATION ACTION PRISE PAR EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action < direction_boundary): 
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step 
            else:
                q_values = model.predict(current_state)[0]
                action = np.argmax(q_values)
                if (action < direction_boundary):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
                
            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            month = new_month + int(timestep / (30*24*60))
            if month >= 12:
                month -= 12
            next_state, reward, game_over = env.update_env(direction, energy_ai, month)
            total_reward += reward
            # STORING THIS NEW TRANSITION INTO THE MEMORY
            transition = [current_state, action, reward, next_state]
            dqn.remember(transition, game_over)
            # GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            # COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
        # PRINTING THE TRAINING RESULTS FOR EACH EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total energy spent with an AI:{:.0f}".format(env.total_energy_ai))
        print("Total energy spent with no AI:{:.0f}".format(env.total_energy_noai))
        # SAVING THE MODEL
        model.save("model.h5")

            
         


