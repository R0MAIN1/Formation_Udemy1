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
#           COURBES D'APPRENTISSAGE & SAUVEGARDE DU RESEAU                  #
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
hist=MonReseau.fit(x=x_train, # données d'entrée pour l'apprentissage
                   y=y_train, # sorties désirées associées aux données d'entrée
                   epochs=10, # nombre de cycles d'apprentissage
                   batch_size=20,
                   validation_data=(x_test,y_test)) # données de test


#----------------------------------------------------------------------------
# GRAPHIQUE pour analyser l'évolution de l'apprentissage
#  => courbes erreurs / fiabilité au cours des cycles d'apprentissage
#----------------------------------------------------------------------------
# création de la figure ('figsize' pour indiquer la taille)
plt.figure(figsize=(8,8))
# evolution du pourcentage des bonnes classifications
plt.subplot(2,1,1)
plt.plot(hist.history['accuracy'],'o-')
plt.plot(hist.history['val_accuracy'],'x-')
plt.title("Taux d'exactitude des prévisions",fontsize=15)
plt.ylabel('Taux exactitude',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='lower right',fontsize=12)
# Evolution des valeurs de l'erreur résiduelle moyenne
plt.subplot(2,1,2)
plt.plot(hist.history['loss'],'o-')
plt.plot(hist.history['val_loss'],'x-')
plt.title('Erreur résiduelle moyenne',fontsize=15)
plt.ylabel('Erreur',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='upper right',fontsize=12)
# espacement entre les 2 figures
plt.tight_layout(h_pad=2.5)
plt.show()


#----------------------------------------------------------------------------
# SAUVEGARDE du réseau après apprentissage
#  => stockage dans un fichier de la description du réseau et des paramètres
#----------------------------------------------------------------------------
# Utilisation de la méthode 'save' de la classe 'tf.keras.Sequential'
# MonReseau.save('MonReseau.h5')
# Utilisation de la fonction 'save_model' du module 'tf.keras.models'
tf.keras.models.save_model(MonReseau, 'MonReseau.h5')
