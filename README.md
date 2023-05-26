# Tracking d'objets dans des vidéos

Auteurs : Lisa Giordani et Olivier Laurent

Dans ce projet, nous avons étudié deux méthodes utilisées pour le suivi d’objets :
- Mean Shift
- Transformée de Hough


## Mean Shift

Le principe général de l'algorithme de Mean Shift est de déterminer comment une distribution de points se déplace dans l'espace, au cours du temps. Cet algorithme est donc utilisé pour suivre le déplacememnt d'objets dans des vidéos. Il peut également être utilisé pour séparer l'arrière-plan statique des objets en mouvement dans une vidéo.

Les résultats obtenus avec cette méthode proviennent des fichiers python suivants :
1. `Tracking_MeanShift_display.py` : code qui permet d’afficher l’histogramme de teinte du model
initial, les histogrammes marginaux, ainsi que les s equences de poids obtenus à partir de la
rétroprojection.
2. `Tracking_MeanShift_updates.py` : code qui permet de modifier la densit ́e calcul ́ee (H : Hue, S :
saturation, V : value) et qui impl ́emente la strat ́egie de mise `a jour de l’histogramme model
choisie.

## Transformée de Hough

Le principe de la transformées de Hough est d'indiquer par une liste de votes des hypothèses de localisation. Ainsi, plus il y a de votes pour une localisation, plus elle est probable. 
La méthode de suivi basée sur les transformées de Hough nécessite donc la construction d'une R-Table. Cette table stocke les votes (vecteurs) des pixels votants classés (en index) en fonction de l'orientation du gradient au niveau du pixel.

Les résultats obtenus avec cette méthode proviennent des fichiers python suivants :
— `Tracking_Hough_display.py` : affiche pour chaque trame l’orientation et le module du gradient,
ainsi que les pixels non votants.
— `Tracking_Hough_compute.py` : calcule la transformée de Hough et localise son maximum.
— `Tracking_Hough_update.py` : utilise différentes méthodes pour calculer le maximum de la transformée de Hough et met à jour la R-Table.



