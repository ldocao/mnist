import pandas as pd

class TestSet:
    def __init__(self, path):
        self.path = path

    def read(self):
        values = pd.read_csv(self.path).values.astype("float32")
        return values