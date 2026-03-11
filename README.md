# SVM from Scratch (with Scikit-Learn Integration)

Ce dépôt contient une implémentation pédagogique d'un **Support Vector Machine (SVM)** construite à partir de zéro en utilisant `numpy`. 

L'objectif est de comprendre les rouages mathématiques de l'optimisation par **Descente de Gradient (Gradient Descent)** tout en intégrant ce modèle dans l'écosystème professionnel de `scikit-learn`.

## 🧠 Concepts Clés
Le modèle apprend à séparer les données en minimisant une fonction de coût basée sur la **Hinge Loss** et la régularisation :

$$J(w, b) = \lambda \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i - b))$$



## 🚀 Fonctionnalités
* **Custom SVM Engine :** Implémentation de `fit` et `predict` basée sur la descente de gradient.
* **Vectorisation :** Utilisation de `numpy` pour des calculs efficaces (calcul de `dw` et `db`).
* **Scikit-Learn Compatibility :** Le modèle est conçu pour fonctionner nativement avec `Pipeline` et `StandardScaler`.
* **Preprocessing :** Démonstration de l'importance de la normalisation des données avant l'entraînement.

## 📂 Structure du projet
```text
├── src/
│   ├── svm.py          # Logique principale du SVM
│   └── implementation.py # Utilisation des outils scikit-learn
└── README.md