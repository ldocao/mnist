import pandas as pd

class Submission:

    def __init__(self, labels):
        self.labels = labels

    def save(self, filename: str) -> None:
        pd.DataFrame({
            "ImageId": list(range(1,len(self.labels)+1)), 
            "Label": self.labels}
            ).to_csv(filename, index=False, header=True)     