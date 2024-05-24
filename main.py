import argparse
import os

from datasets import load_dataset
from dotenv import load_dotenv
from together import Together
from werkzeug.debug import Console

from Datamanager import DataManager
from Transformer import Transformer
from jsonifier import Jsonifier
from textifier import Textifier

class TogetherAPI:
    @staticmethod
    def create_client():
        load_dotenv()
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        return client
    @staticmethod
    def upload_file(file_path):
        client = TogetherAPI.create_client()
        resp = client.files.upload(file=file_path) # uploads a file
        print(resp.dict())
    @staticmethod
    def file_uploaded():
        client = TogetherAPI.create_client()
        filesUploaded = client.files.list() # lists all uploaded files
        print(filesUploaded)
    @staticmethod
    def tuning(model, training_file, n_epochs, n_checkpoints, batch_size, learning_rate):
        client = TogetherAPI.create_client()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        resp = client.fine_tuning.create(
            training_file=training_file, #example 'file-29713245728'
            model=model, # example 'meta-llama/Meta-Llama-3-8B-Instruct'
            n_epochs=n_epochs,
            n_checkpoints=n_checkpoints,
            batch_size=batch_size,
            learning_rate=learning_rate, # example 1e-5
            wandb_api_key=wandb_api_key,
        )
        print(resp)

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='A transformer for your dataset in order to do a fine-tuning process')
    argument_parser.add_argument('--genTrainer', type=str, default=None, nargs=3,
                                 metavar=('Name_Dataset', 'TypeOfPrompt', 'TypeOfFileGenerated'),
                                 help='')
    args = argument_parser.parse_args()
    console = Console()
    Name_Dataset, TypeOfPrompt, TypeOfFileGenerated = args.genTrainer
    dataset = load_dataset(Name_Dataset)
    datamanager = DataManager(dataset)
    if TypeOfPrompt.upper() == "ALPACA":
        transformed_dataset = datamanager.transform_dataset_for_Alpaca()
    elif TypeOfPrompt.upper() == "CHATML":
        transformed_dataset = datamanager.transform_dataset_for_ChatML()
    else:
        print("Please insert a valid type of prompt")
    if TypeOfFileGenerated.upper() == "JSONL":
        transformers = Jsonifier(transformed_dataset)
    elif TypeOfFileGenerated.upper() == "TXT":
        transformers = Textifier(transformed_dataset)
    else:
        print("Please insert a valid type of prompt")
    transformers.transform()



    #TogetherAPI.upload_file(datamanager.dataset)
    #TogetherAPI.file_uploaded()
    #TogetherAPI.tuning("model.gguf", "trainer.txt", 10, 5, 8, 1e-5)

