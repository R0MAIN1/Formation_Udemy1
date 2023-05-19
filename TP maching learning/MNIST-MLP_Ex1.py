# -*- coding: utf-8 -*-

#---------------------------------------------------------
# Architecture réseau : MLP (50,10)
# Nb itérations       : 10
# Fct activation      : relu, softmax (sortie)
# Mesure erreurs      : categorical_crossentropy
# Algo apprentissage  : adam
# Utilisation dropout : non
# Autre               : RAS
#---------------------------------------------------------
# Taux apprentissage  : ~98,7 %
# taux validation     : ~97 %
#---------------------------------------------------------

#############################################################################
#               FONCTIONNEMENT DE BASE DU RESEAU                            #
#############################################################################

# Importation des modules TensorFlow & Keras
#  => construction et exploitation de réseaux de neurones
import tensorflow as tf

# Importation du module numpy
#  => manipulation de tableaux multidimensionnels
import numpy as np

# Importation du module graphique
#  => tracé de courbes et diagrammes
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Chargement des données d'apprentissage et de tests
#----------------------------------------------------------------------------
# Chargement en mémoire de la base de données des caractères MNIST
#  => tableaux de type ndarray (Numpy) avec des valeur entières
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#----------------------------------------------------------------------------
# Changements de format pour exploitation
#----------------------------------------------------------------------------
# les valeurs associées aux pixels sont des entiers entre 0 et 255
#  => transformation en valeurs réelles entre 0.0 et 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0
# Les données en entrée sont des matrices de pixels 28x28
#  => transformation en vecteurs de 28x28=784 pixels
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
# Les données de sortie sont des entiers associés aux chiffres à identifier
#  => transformation en vecteurs booléens pour une classification en 10 valeurs
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)


#----------------------------------------------------------------------------
# DESCRIPTION du modèle Perceptron multicouches (MLP)
#  => 1 couche cachée avec 50 neurones
#----------------------------------------------------------------------------
# Création d'un réseau multicouches
MonReseau = tf.keras.Sequential()
# Description de la 1ère couche cachée
MonReseau.add(tf.keras.layers.Dense(
        units=50,              # 50 neurones
        input_shape=(784,),    # nombre d'entrées (car c'est la 1ère couche)
        activation='relu'))    # fonction d'activation

# Description de la couche de sortie
MonReseau.add(tf.keras.layers.Dense(
        units=10,              # 10 neurones
        activation='softmax')) # fonction d'activation (sorties sur [0,1])


#----------------------------------------------------------------------------
# COMPILATION du réseau 
#  => configuration de la procédure pour l'apprentissage
#----------------------------------------------------------------------------
MonReseau.compile(optimizer='adam',                # algo d'apprentissage
                  loss='categorical_crossentropy', # mesure de l'erreur
                  metrics=['accuracy'])            # mesure du taux de succès


#----------------------------------------------------------------------------
# APPRENTISSAGE du réseau
#  => calcul des paramètres du réseau à partir des exemples
#----------------------------------------------------------------------------
MonReseau.fit(x=x_train, # données d'entrée pour l'apprentissage
              y=y_train, # sorties désirées associées aux données d'entrée
              batch_size=20,
              epochs=10) # nombre de cycles d'apprentissage 


#----------------------------------------------------------------------------
# EVALUATION de la capacité à généraliser du réseau
#  => test du réseau sur des exemples non utilisés pour l'apprentissage 
#----------------------------------------------------------------------------
print()
perf=MonReseau.evaluate(x=x_test, # données d'entrée pour le test
                        y=y_test) # sorties désirées pour le test
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))
