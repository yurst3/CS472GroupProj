from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np

domain = np.arange(1,17,2)
bag_acc = np.load('300_bag_accuracies.npy')*100

plt.plot(domain, bag_acc[0])
#plt.axes((0,15,60,70))
plt.title('Prediction Accuracy vs Number of Models')
plt.xlabel('Models')
plt.ylabel('Prediction Accuracy %')
plt.savefig('300_bag_accuracy.png')