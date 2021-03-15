import random
import pandas as pd

from submission import Submission

class RandomModel:
    def __init__(self, test):
        self.test = test

    def predict(self):
        MAX_DIGIT = 9
        values = {}
        for i in self.test.index:
            values[i] = random.randrange(MAX_DIGIT-1)
        predictions = pd.DataFrame.from_dict(values, columns=[Submission.VALUE_NAME], orient="index")
        predictions = predictions[Submission.VALUE_NAME] 
        predictions.index.name = Submission.INDEX_NAME
        return predictions


