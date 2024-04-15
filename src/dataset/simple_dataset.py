from src.dataset.dataset import Dataset

class SimpleDataset(Dataset):

    # dataset --- [(x,y)... ]
    def __init__(self, dataset=None):
        if dataset is None: raise Exception("No data provided")
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]