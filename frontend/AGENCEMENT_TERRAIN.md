📘 Note technique

Placement des hexagones pour un terrain isométrique « type Polytopia »

1. Dimensions de base

On prend un hexagone régulier pointu vers le haut, de côté a.

	•	Largeur : W = √3 · a

	•	Hauteur totale : H = 2a

	•	Hauteur du losange supérieur (partie qui sert de recouvrement) : H/4 = a/2

2. Espacing entre tuiles

Pour obtenir un terrain continu :

	•	Espacement horizontal :

dx = √3 · a

	•	Espacement vertical :

dy = (3/2) · a

Cet espacement vertical provoque exactement un recouvrement de :

H - dy = a/2

C'est la hauteur du losange supérieur.

C'est ce chevauchement qui donne l'herbe uniforme.

3. Placement sur la grille

Pour une grille indexée (i, j) :

x = √3 · a · (i + 0.5 · (j mod 2))

y = (3/2) · a · j

	•	Chaque ligne est décalée d'un demi-hexagone à droite.

	•	L'ordre d'affichage doit aller du haut vers le bas pour gérer les recouvrements.

⸻

Résultat

En appliquant strictement ces proportions :

	•	Les hexagones se placent sans trous.

	•	Le sommet en losange de chaque hexagone recouvre exactement le bas du suivant.

	•	L'herbe apparaît d'un seul tenant, comme dans Polytopia.



