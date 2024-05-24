from datasets import DatasetDict

from Transformer import Transformer


class DataManager:
    def __init__(self, datasets: DatasetDict):
        self.dataset = datasets

    def transform_dataset_for_Alpaca(self):
        instructions = self.dataset["train"]["instruction"]
        inputs = self.dataset["train"]["input"]
        outputs = self.dataset["train"]["output"]
        transformed_dataset = []
        for i in range(len(instructions)):
            system_header = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            instruction = f"<|start_header_id|>{instructions[i]}<|end_header_id|>\n"
            user_header = "<|start_header_id|>user<|end_header_id|>\n"
            user_msg_1 = f"<|start_header_id|>{inputs[i]}<|end_header_id|>\n"
            assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"
            model_answer = f"<|start_header_id|>{outputs[i]}<|end_header_id|>\n"
            transformed_example = f"{system_header}{instruction}{user_header}{user_msg_1}{assistant_header}{model_answer}"
            transformed_dataset.append(transformed_example)

        return transformed_dataset

    def transform_dataset_for_ChatML(self):
        instructions = self.dataset["train"]["instruction"]
        outputs = self.dataset["train"]["output"]
        transformed_dataset = []
        for i in range(len(instructions)):
            system_header = "<s><|im_start|>system\n"
            instruction = (f"This is a conversation with your helpful Coding assistant."
                           f" Assistant can generate Code in various Programming Languages along with necessary explanation.<|im_end|>\n")
            user_header = "<|im_start|>user\n"
            prompt = f"<{instructions[i]}<|im_end|>\n"
            model_answer = f"<|im_start|>assistant{outputs[i]}\n"
            transformed_example = f"{system_header}{instruction}{user_header}{prompt}{model_answer}"
            transformed_dataset.append(transformed_example)
        return transformed_dataset
