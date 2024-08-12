
"""
KServe inference script for NLLB-200 translation model.
"""

import argparse
import os
from typing import List
from kserve import (InferOutput, InferRequest, InferResponse, Model, ModelServer, model_server)
from kserve.utils.utils import generate_uuid
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ctranslate2
import sentencepiece as spm

# Constants
MODEL_DIR = "./saved_model/checkpoint-1700"

class TranslationModel(Model):
    """
    KServe inference implementation of NLLB-200 translation model.
    """

    def __init__(self, name: str):
        """
        Initialize the translation model.
        Args:
            name (str): Name of the model.
        """        
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        #self.tokenizer = None
        self.sp_source_model = None
        self.sp_target_model = None
        self.load()

    def load(self) -> None:
        """
        Load model and tokenizer from disk.
        """
        try:
            self.sp_source_model = spm.SentencePieceProcessor(model_file=MODEL_DIR+'/source.spm')
            self.sp_target_model = spm.SentencePieceProcessor(model_file=MODEL_DIR+'/target.spm')
            #self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            self.model = ctranslate2.Translator(MODEL_DIR)
            print('Model and tokenizer loaded')
            self.ready = True
        except Exception as e:
            print('Error loading model: ', e)
            self.ready = False

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> str:
        """
        Preprocess inference request.

        Args:
            payload (InferRequest): The input payload containing the text to translate.

        Returns:
            str: Preprocessed text ready for translation.
        """
        return payload.inputs[0].data[0].lower()

    def predict(self, data: str, *args, **kwargs) -> InferResponse:
        """
        Make prediction using the model.
        Args:
            data (str): Preprocessed input text.

        Returns:
            InferResponse: KServe inference response containing the translated text.
        """
        source_sentences = [data.strip()]
        print(source_sentences)
        translation = self._translate(self.model, source_sentences)[0]

        return self._create_response(translation)
    

    # Ctranslate2 translation
    def _translate(self, model, text):
        tokens = self.sp_source_model.encode(text, out_type=str)
        tokens[0].insert(0,"dyu")
        tokens[0].append("</s>")
        tokens[0].append("fr")
        # tokens = ["dyu"] + [[t] for t in tokens] + ["</s>"] + ["fr"]
        try:
            results = model.translate_batch(tokens)
            # The translated results are token strings, so we need to convert them to IDs before decoding
            translations = []
            for translation in results:
                # Convert token strings to IDs before decoding
                decoded_text = self.sp_target_model.decode(translation.hypotheses[0])
                translations.append(decoded_text)
        except Exception as e:
            print(f"Translation error: ", e)
            translations = [""]  # Return empty string if translation fails
        translations = ["some thing"] 
        return translations

    def _create_response(self, translation: str) -> InferResponse:
        """
        Create InferResponse object.

        Args:
            translation (str): Translated text.

        Returns:
            InferResponse: KServe inference response object.
        """
        return InferResponse(
            model_name=self.name,
            infer_outputs=[InferOutput(name="output-0", shape=[1], datatype="STR", data=[translation])],
            response_id=generate_uuid()
        )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    # Check if '--model_name' is already defined
    model_name_defined = any('--model_name' in action.option_strings for action in model_server.parser._actions)

    if not model_name_defined:
        model_server.parser.add_argument(
            '--model_name', 
            default='model', 
            help='The name that the model is served under.'
        )
    return parser.parse_args()

def main():
    """
    Main function to start the model server.
    """
    args = parse_arguments()
    model = TranslationModel(args.model_name)
    ModelServer().start([model])

if __name__ == "__main__":
    main()

