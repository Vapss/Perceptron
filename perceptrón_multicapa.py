from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
print ('Cargando datos...')
df = pd.read_csv('train.csv', sep=',', engine='python')
X = df.drop(['label'],axis=1).values   
y = df['label'].values

#Separa el corpus cargado en el DataFrame en el 70% para entrenamiento y el 30% para pruebas
print ('Separando los conjuntos de datos...')
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, shuffle = True, random_state=0)

#~ for i in range(10):
    #~ img = X_train[i].reshape((28,28))
    #~ plt.imshow(img, cmap="Greys")
    #~ plt.show()

	
clf = MLPClassifier(hidden_layer_sizes=(40), max_iter=10, random_state = 0)

print ('Entrenando red neuronal ...')
clf.fit(X_train, y_train)

print ('Predicción de la red neuronal')
y_pred = clf.predict(X_test)

print (classification_report(y_test, y_pred))
print("Training set score: %f" % clf.score(X_train, y_train))

for i in range(10):
    img = X_test[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.title('Real: ' + str(y_test[i]) + ' Predicted: ' + str(y_pred[i]), fontsize = 20);
    plt.show()


    

