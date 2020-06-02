from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("ms_note_predictor")
class ms_note_predictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.
    Registered as a `Predictor` with name "text_classifier".
    """

    def predict(self, tokenized_note: str, patient_id: int) -> JsonDict:
        return self.predict_json({"tokenized_note": tokenized_note, "patient_id": patient_id})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"tokenized_note": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        tokenized_note = json_dict["tokenized_note"]
        patient_id = json_dict["patient_id"]
        return self._dataset_reader.text_to_instance(tokenized_note, patient_id)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]

