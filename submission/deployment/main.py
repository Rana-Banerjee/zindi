
"""
KServe inference script for NLLB-200 translation model.
"""

import argparse
import os
from typing import List
from kserve import (InferOutput, InferRequest, InferResponse, Model, ModelServer, model_server)
from kserve.utils.utils import generate_uuid
import ctranslate2
import sentencepiece as spm

# Constants
MODEL_DIR = "./saved_model/checkpoint-618"

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
        self.tokenizer = None
        self.mpn = None
        self.load()

    def load(self) -> None:
        """
        Load model and tokenizer from disk.
        """
        try:
            self.tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(MODEL_DIR, 'sentencepiece.bpe.model'))
            self.model = ctranslate2.Translator(
                MODEL_DIR,
                device="cpu",
                device_index=0,
                compute_type="int8",
                intra_threads=os.cpu_count(),
                inter_threads=1,
            )
            print('Model and tokenizer loaded')
            self.ready = True
        except Exception as e:
            print('Error loading model:', e)
            self.ready = False

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> str:
        """
        Preprocess inference request.

        Args:
            payload (InferRequest): The input payload containing the text to translate.

        Returns:
            str: Preprocessed text ready for translation.
        """
        text = payload.inputs[0].data[0]
        return text.strip()

    def predict(self, data: str, *args, **kwargs) -> InferResponse:
        """
        Make prediction using the model.

        Args:
            data (str): Preprocessed input text.

        Returns:
            InferResponse: KServe inference response containing the translated text.
        """
        translation = self._translate(self.model, data)
        return self._create_response(translation)

    def _translate(self, model, text):
        """
        Translate the input text using the ctranslate2 library.

        Args:
            model (ctranslate2.Translator): The translation model.
            text (str): The input text to be translated.

        Returns:
            str: The translated text.
        """
        target_prefix = [['fra_Latn']] * len([text])
        source_sents_subworded = [["dyu_Latn"] + self.tokenizer.encode_as_pieces(sent) + ["</s>"] for sent in [text]]
        try:
            translations = self.model.translate_batch(
                source_sents_subworded,
                batch_type="tokens",
                max_batch_size=256,
                beam_size=1,
                target_prefix=target_prefix,
                return_scores=False,
                return_attention=False,
                return_alternatives=False,
            )
            translation = translations[0].hypotheses[0]
            if "fra_Latn" in translation:
                translation.remove("fra_Latn")
            trans = self.tokenizer.decode(translation)
        except Exception as e:
            trans = ["Error: " + str(e)]  # Return error message if translation fails
        return trans

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
