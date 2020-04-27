from typing import List, Dict, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("grammar_parser_predictor")
class Seq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    [`ComposedSeq2Seq`](../models/encoder_decoders/composed_seq2seq.md) and
    [`SimpleSeq2Seq`](../models/encoder_decoders/simple_seq2seq.md) and
    [`CopyNetSeq2Seq`](../models/encoder_decoders/copynet_seq2seq.md).
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use

        query_id = outputs["query_id"]
        question = outputs["question"]
        question_tokens = outputs["question_tokens"]
        predicted_logical_form = outputs["predicted_logical_form"]
        gold_logical_form = outputs.get("gold_logical_form", "None")
        exact_match = outputs.get("exact_match", None)

        output_str = f"query_id: {query_id}\n" \
                     f"question: {question}\n" \
                     f"gold-program: {gold_logical_form}\n" \
                     f"pred-program: {predicted_logical_form}\n" \
                     f"exact-match: {exact_match}\n" \
                     f" --------------------------------- \n"

        return output_str

