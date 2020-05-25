from typing import Dict, List, Tuple, Any
import logging
import json
import random
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, MetadataField, ListField, IndexField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp_semparse.fields.production_rule_field import ProductionRuleField

from qdmr.data.utils import Node, nested_expression_to_lisp, read_qdmr_json_to_examples, QDMRExample
from qdmr.domain_languages.qdmr_language import QDMRLanguage
from qdmr.domain_languages.drop_language import DROPLanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("qdmr_grammar_reader")
class GrammarDatasetReader(DatasetReader):
    """ Read a json file containing DROP programs obtained from parsing QDMR data.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the target sequence.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    source_add_start_token : bool, (optional, default=False)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 random_seed: int = 0) -> None:
        super().__init__(lazy)
        self._random_seed = random_seed
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer(namespace="source_tokens")}
        # Number of examples with no gold program
        self.skipped_wo_program: int = 0
        self.longest_program: int = 0


    @overrides
    def _read(self, file_path: str):
        """
        This dataset reader consumes the data from
        https://github.com/jkkummerfeld/text2sql-data/tree/master/data
        formatted using ``scripts/reformat_text2sql_data.py``.

        Parameters
        ----------
        file_path : ``str``, required.
            For this dataset reader, file_path can either be a path to a file `or` a
            path to a directory containing json files. The reason for this is because
            some of the text2sql datasets require cross validation, which means they are split
            up into many small files, for which you only want to exclude one.
        """

        logger.info("Reading file at %s", file_path)
        qdmr_examples: List[QDMRExample] = read_qdmr_json_to_examples(file_path)


        for qdmr_example in qdmr_examples:
            query_id: str = qdmr_example.query_id
            question: str = qdmr_example.question
            program_tree: Node = qdmr_example.program_tree   # None if absent
            if not program_tree:
                self.skipped_wo_program += 1
                continue

            # qdmr_language: QDMRLanguage = QDMRLanguage()
            drop_language: DROPLanguage = DROPLanguage()
            logical_form = nested_expression_to_lisp(program_tree.get_nested_expression())
            gold_action_sequence: List[str] = drop_language.logical_form_to_action_sequence(logical_form)

            extras: Dict = qdmr_example.extras

            additional_metadata = {"query_id": query_id, "question": question, "logical_form": logical_form}

            # TODO(nitish): Some utterance gives Spacy error even though SpacyTokenizer is able to parse it cmd shell
            # try:
            instance = self.text_to_instance(utterance=question,
                                             language=drop_language,
                                             extras=extras,
                                             gold_action_sequence=gold_action_sequence,
                                             additional_metadata=additional_metadata)
            if instance is not None:
                yield instance

        logger.info(f"Total examples: {len(qdmr_examples)}  "
                    f"Skipped w/o gold-program: {self.skipped_wo_program}  "
                    f"Longest program: {self.longest_program}")
        # reset
        self.skipped_wo_program = 0
        self.longest_program = 0

    @overrides
    def text_to_instance(self,
                         utterance: str,
                         language: DROPLanguage,
                         extras: Dict,
                         gold_action_sequence: List[str] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        if "question_tokens" in extras:
            question_tokens: List[str] = extras["question_tokens"]
            tokenized_source: List[Token] = [Token(t) for t in question_tokens]
        else:
            logger.info("Tokenizing questions. Earlier it presented errors!! Pre-tokenize utterances ")
            tokenized_source: List[Token] = self._source_tokenizer.tokenize(utterance)

        source_field = TextField(tokenized_source, self._source_token_indexers)
        fields["tokens"] = source_field

        metadata = {
            "utterance_tokens": [token.text for token in tokenized_source]
        }

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            rule_field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(rule_field)
        action_field = ListField(production_rule_fields)
        fields["valid_actions"] = action_field

        action_map = {
            action.rule: i  # type: ignore
            for i, action in enumerate(action_field.field_list)
        }

        fields["languages"] = MetadataField(language)

        if gold_action_sequence:
            index_fields: List[IndexField] = []
            for production_rule in gold_action_sequence:
                index_fields.append(IndexField(action_map[production_rule], action_field))
            if not gold_action_sequence:
                index_fields = [IndexField(-1, action_field)]

            action_sequence_field = ListField(index_fields)
            fields["action_sequence"] = action_sequence_field
            self.longest_program = max(len(gold_action_sequence), self.longest_program)

            # If target is given, provide attention-supervision is available
            fastalign_sup_key = "fastalign.grammar"
            if fastalign_sup_key in extras:
                # For each output-token, list of input tokens to attend to
                output_input_alignments: List[List[int]] = extras[fastalign_sup_key]
                assert len(output_input_alignments) == len(gold_action_sequence)
                # Since while decoding, we'll only supervise attention for decoding of program-tokens and not END
                attention_matrix = np.zeros((len(output_input_alignments), len(tokenized_source)), dtype=np.float32)
                for output_idx, input_idxs in enumerate(output_input_alignments):
                    attention_matrix[output_idx, input_idxs] = 1.0
                fields["attention_supervision"] = ArrayField(attention_matrix, padding_value=0)

        metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
