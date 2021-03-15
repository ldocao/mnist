import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from test_set import TestSet
from train_set import TrainSet

from submission import Submission
from pixels import Pixel
from random_model import RandomModel
from mlp import MultiLayerPerceptron


TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"
test_set = TestSet(TEST_PATH).read()
train_set = TrainSet(TRAIN_PATH).read()
x_train = train_set.drop("label", axis=1).values.astype('float32')
y_train = train_set["label"].values.astype('int32')

#preprocessing
max_value = np.max(x_train)
mean_value = np.mean(x_train)
test_set = (test_set - mean_value) / max_value
x_train = (x_train - mean_value) / max_value


#MLP
MLP_SUBMISSION = "mlp.csv"
mlp = MultiLayerPerceptron(x_train, y_train)
mlp.fit()
predictions = mlp.predict(test_set)
Submission(predictions).save(MLP_SUBMISSION)


for i in range(1, 10):
    random_image = test_set[i,:]
    print(predictions[i])
    Pixel(random_image).display()

