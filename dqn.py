#Mise en œuvre d’un algorithme de Deep Q-Learning intégrant
#le modèle de régression linéaire multiple pour l’optimisation 2nergétique des Data Center
# Implementing Deep Q-Learning with Experience Replay# Implementtion de l'algo deep q learning

# Importing the libraries# Import des libraries
import numpy as np

class DQN(object):
    #INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE DQN# INITIALISATION DE LA CLASSE
    def __init__(self, max_memory=100, discount=0.9):
        #self.memory = list() 
        self.max_memory = max_memory 
        self.discount = discount
        self.memory = list() 
    # MAKING A METHOD THAT BUILDS THE MEMORY IN EXPERIENCE REPLAY# METHODE POUR LE PRECESSUS DE REMPLISSAGE DE L'EXPERIENCE REPLAY
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]#on supprime le tout premier element de la memoire
    # MAKING A METHOD THAT BUILDS TWO BATCHES OF INPUTS AND TARGE# CREATION OU OBTENTION DE DEUX BACHES(ENTRE ET LA CIBLE)
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory) 
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        #inputs = np.zeros((batch_size, num_inputs))
        #targets = np.zeros((batch_size, num_outputs))
        inputs = np.zeros((min(len_memory, batch_size), num_inputs)) 
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        transitions = np.random.randint(0, len_memory,size = min(len_memory, batch_size))
        for i, idx in enumerate(transitions):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            #targets[i, action] = model.predict(current_state)[0]
            targets[i, action] = reward + self.discount * np.max(model.predict(next_state)[0])# equation de Belman
            #Q_sa = np.max(model.predict(next_state)[0])
            if game_over==1:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * np.max(model.predict(next_state)[0])
        return inputs, targets


