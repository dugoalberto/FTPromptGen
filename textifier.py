from Transformer import Transformer


class Textifier(Transformer):
    def __init__(self, dataset):
        super().__init__(dataset)
    def transform(self):
        with open("trainer.txt", "w") as file:
            for data in self.dataset:
                file.write(data)
