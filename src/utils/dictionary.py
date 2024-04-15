class DictionaryMask():
    def __init__(self, item):
        self.placeholder = item

    def __getitem__(self, item):
        return self.placeholder
