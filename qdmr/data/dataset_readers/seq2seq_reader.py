from typing import Dict, List, Tuple, Any
import logging
import json
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import qdmr.data.utils as qdmr_utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("qdmr_seq2seq_reader")
class Seq2SeqDatasetReader(DatasetReader):
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
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = False,
                 lazy: bool = False,
                 random_seed: int = 0) -> None:
        super().__init__(lazy)
        self._random_seed = random_seed
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or SpacyTokenizer(split_on_spaces=True)
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer(namespace="source_tokens")}
        self._target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer(namespace="target_tokens")}
        self._source_add_start_token = source_add_start_token
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
        qdmr_examples: List[qdmr_utils.QDMRExample] = qdmr_utils.read_qdmr_json_to_examples(file_path)

        for qdmr_example in qdmr_examples:
            query_id: str = qdmr_example.query_id
            question: str = qdmr_example.question
            typed_masked_nested_expression: List = qdmr_example.typed_masked_nested_expr
            if not typed_masked_nested_expression:
                self.skipped_wo_program += 1
                continue

            linearized_program: List[str] = qdmr_utils.nested_expression_to_linearized_list(
                typed_masked_nested_expression)
            linearized_program_str = " ".join(linearized_program)

            additional_metadata = {"query_id": query_id, "question": question}

            # TODO(nitish): Some utterance gives Spacy error even though SpacyTokenizer is able to parse it cmd shell
            try:
                instance = self.text_to_instance(utterance=question,
                                                 target_string=linearized_program_str,
                                                 additional_metadata=additional_metadata)
            except:
                self.skipped_wo_program += 1
                continue
            if instance is not None:
                yield instance

        logger.info(f"Total examples: {len(qdmr_examples)}  "
                    f"Skipped w/o gold-program: {self.skipped_wo_program}  "
                    f"Longest program: {self.longest_program}")

    @overrides
    def text_to_instance(self,
                         utterance: str,
                         target_string: str = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        tokenized_source: List[Token] = self._source_tokenizer.tokenize(utterance)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
            tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        fields["source_tokens"] = source_field

        metadata = {
            "utterance_tokens": [token.text for token in tokenized_source]
        }

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            # Definitely add START and END symbols to target
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields["target_tokens"] = target_field
            metadata.update({"target_tokens": [token.text for token in tokenized_target]})
            self.longest_program = len(tokenized_target) if len(tokenized_target) > self.longest_program \
                else self.longest_program

        metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)