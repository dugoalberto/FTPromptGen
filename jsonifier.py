import jsonlines

from Transformer import Transformer


class Jsonifier(Transformer):
    def __init__(self, dataset):
        super().__init__(dataset)
    def transform(self):
        output_file_path = "transformed_dataset.jsonl"
        # Write the transformed result to the JSONL file
        with jsonlines.open(output_file_path, mode="w") as writer:
            for item in self.dataset[::2]:
                writer.write({"text": item})

        print(f"Saved transformed data to {output_file_path}")
