import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles

# 1. Création d'un dataset "cercles" (2 features, non linéairement séparable)
# Les points sont répartis en deux classes : un cercle intérieur et un anneau extérieur
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)



# 2. Entraînement du modèle SVM avec un noyau RBF
# Le kernel 'rbf' permet de projeter ces données 2D dans un espace plus haut
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)

# 3. Visualisation de la frontière de décision
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('SVM Decision Boundary in 2D')
plt.show()







