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
#                  CHARGEMENT & EXPLOITATION D'UN RESEAU                    #
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
# Changements de format pour exploitation des données de test
#----------------------------------------------------------------------------
# les valeurs associées aux pixels sont des entiers entre 0 et 255
#  => transformation en valeurs réelles entre 0.0 et 1.0
x_test = x_test / 255.0
# Les données en entrée sont des matrices de pixels 28x28
#  => transformation en vecteurs de 28x28=784 pixels
x_test  = x_test.reshape(10000, 784)
# Les données de sortie sont des entiers associés aux chiffres à identifier
#  => transformation en vecteurs booléens pour une classification en 10 valeurs
y_test  = tf.keras.utils.to_categorical(y_test, 10)


#----------------------------------------------------------------------------
# CHARGEMENT d'un réseau depuis un fichier
#  => pour exploiter le réseau avec les paramètres calculés par apprentissage 
#----------------------------------------------------------------------------
MonReseau=tf.keras.models.load_model('MonReseau.h5')
# Affichage des caractéristiques du réseau
print('\nCARACTERISTIQUES DU RESEAUX:')
print('==============================')
MonReseau.summary()


#----------------------------------------------------------------------------
# EXPLOITATION du réseau
#  => calcul des sorties associées à une image transmise en entrée
#----------------------------------------------------------------------------
# N° de l'exemple à tester dans la base de tests
i=25
# y_test est un tableau de booléens avec une seule valeur à 1
# => argmax() retourne la position du 1 qui correspond au chiffre cherché
print('\nCALCUL DES SORTIES ASSOCIEES A UNE ENTREE:')
print('============================================')
print("Test N°{} => chiffre attendu {}".format(i,y_test[i].argmax()))
print("Résultat du réseau :")
# Utilisation des fonctions "predict" associées à l'objet MonReseau
#  => Entrée: un tableau de vecteurs (ici un seul vecteur x_test[i:i+1])
#  => Sortie: un tableau avec les sorties pour chaque vecteur d'entrée
print("Sorties brutes:",  MonReseau.predict(x_test[i:i+1])[0])
#print("Classe de sortie:",MonReseau.predict_classes(x_test[i:i+1])[0],'\n')
print("Classe de sortie:",MonReseau.predict(x_test[i:i+1])[0].argmax(),'\n')



#----------------------------------------------------------------------------
# EXPLOITATION du réseau
#  => affichage des exemples de caractères bien et mal reconnus
#----------------------------------------------------------------------------
print('FIABILITE DU RESEAU:')
print('====================')
# Résultat du réseau avec des données de tests
perf=MonReseau.evaluate(x=x_test, # données d'entrée pour le test
                        y=y_test) # sorties désirées pour le test
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))
NbErreurs=int(10000*(1-perf[1]))
print("==>",NbErreurs," erreurs de classification !")
print("==>",10000-NbErreurs," bonnes classifications !")
# Calcul des prédictions du réseaux pour l'ensemble des données de test
Predictions=MonReseau.predict(x_test)
# Affichage des caractères bien/mal reconnus avec une matrice d'images
i=-1
Couleur='Red' # à remplacer par 'Green' pour les bonnes reconnaissances
plt.figure(figsize=(12,8), dpi=200)
for NoImage in range(12*8):
    i=i+1
    # '!=' pour les bonnes reconnaissances, '==' pour les erreurs
    while y_test[i].argmax() == Predictions[i].argmax(): i=i+1
    plt.subplot(8,12,NoImage+1)
    # affichage d'une image de digit, en format niveau de gris
    plt.imshow(x_test[i].reshape(28,28), cmap='Greys', interpolation='none')
    # affichage du titre (utilisatin de la méthode format du type str)
    plt.title("Prédit:{} - Correct:{}".format(MonReseau.predict(
                        x_test[i:i+1])[0].argmax(),y_test[i].argmax()),
                        pad=2,size=5, color=Couleur)
    # suppression des graduations sur les axes X et Y
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
# Affichage de la figure
plt.show()
