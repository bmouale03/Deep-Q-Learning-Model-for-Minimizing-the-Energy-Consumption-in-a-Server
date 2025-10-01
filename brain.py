#Mise en œuvre d’un algorithme de Deep Q-Learning intégrant
#le modèle de régression linéaire multiple pour l’optimisation 2nergétique des Data Center
# Building the Brain# Developpement du cerveau

# Importing the libraries# Import des librairies necessaires
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam # un optimizeur
#from keras.src import tree
#from keras.src import optree

# BUILDING THE BRAIN# CREATION DU CERVEAU
class Brain(object):
    def __init__(self, learning_rate=0.001, number_actions=5):
        self.learning_rate = learning_rate
        # BUILDIND THE INPUT LAYER COMPOSED OF THE INPUT STATE# CREATION DE LA  COUCHE D'ENTREE
        states = Input(shape = (3,))
        # BUILDING THE FULLY CONNECTED HIDDEN LAYERS# LES COUCHES CACHEES
        x = Dense(units = 64, activation = 'sigmoid')(states) 
        x = Dropout(rate = 0.1)(x)
        y = Dense(units = 32, activation = 'sigmoid')(x)
        y = Dropout(rate = 0.1)(y)
        # BUILDING THE OUTPUT LAYER, FULLY CONNECTED TO THE LAST HIDDEN LAYER# CREATION DE LA  COUCHE DE SORTIE
        q_values = Dense(units = number_actions, activation = 'softmax')(y)# Sofmax est beaucoup plus utiliser pour la classification car elle permet d'obtenir les probabilités
        # ASSEMBLING THE FULL ARCHITECTURE INSIDE A MODEL OBJECT#
        self.model = Model(inputs = states, outputs = q_values)# CREATION DU MODELE
        # COMPILING THE MODEL WITH A MEAN-SQUARED ERROR LOSS AND A CHOSEN OPTIMIZER# COMPILATION DU MODELE
        self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))# mse pour les problemes de regressions
        
        