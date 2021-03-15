import pandas as pd

class TrainSet:
    def __init__(self, path: str):
        self.path = path

    def read(self):
        values = pd.read_csv(self.path)
        return values