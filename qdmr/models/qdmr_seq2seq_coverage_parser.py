from typing import Dict, List, Tuple, Iterable, Any

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules import Dropout

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU, Average

from qdmr.modules.attention.coverage_attention import CoverageAdditiveAttention

"""
Copy from (github.com/allenai/allennlp-models/blob/master/allennlp_models/seq2seq/simple_seq2seq_model.py)
Two reasons for copying -- can modify as we like and no dependence on the allennlp-models codebase
"""


@Model.register("qdmr_seq2seq_coverage")
class SimpleSeq2SeqCoverage(Model):
    """
    This `SimpleSeq2Seq` class is a `Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : `int`
        Maximum length of decoded sequences.
    target_namespace : `str`, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : `int`, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : `Attention`, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    beam_size : `int`, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : `float`, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015](https://arxiv.org/abs/1506.03099).
    use_bleu : `bool`, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    ngram_weights : `Iterable[float]`, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        max_decoding_steps: int,
        use_attention_loss: bool = False,
        beam_size: int = None,
        target_namespace: str = "tokens",
        target_embedding_dim: int = None,
        dropout: float = 0.2,
        scheduled_sampling_ratio: float = 0.0,
        use_bleu: bool = True,
        bleu_ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._dropout = Dropout(p=dropout)

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(
                self.vocab._padding_token, self._target_namespace
            )
            self._bleu = BLEU(
                bleu_ngram_weights, exclude_indices={pad_index, self._end_index, self._start_index}
            )
        else:
            self._bleu = None

        self._exact_match = Average()

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        self._attention = CoverageAdditiveAttention(self._encoder.get_output_dim(),
                                                    self._encoder.get_output_dim())

        self._coverage_loss = Average()

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._target_embedder = Embedding(
            num_embeddings=num_classes, embedding_dim=target_embedding_dim
        )

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
            self.use_attention_loss = use_attention_loss
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim
            self.use_attention_loss = False

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        _, output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        question_spans: torch.LongTensor = None,
        target_tokens: TextFieldTensors = None,
        attention_supervision: torch.Tensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass with decoder logic for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : `TextFieldTensors`, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        # Returns

        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens)

        if target_tokens:
            state = self._init_decoder_state(state)
            if not self.use_attention_loss:
                attention_supervision = None

            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens, attention_supervision)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            exact_matches = None
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"]["tokens"])
                # Using 2nd token and beyond for gold since the prediction does not contain @start@ token
                exact_matches = self.exact_match_sequences(predicted_indices=best_predictions,
                                                           gold_indices=target_tokens["tokens"]["tokens"][:, 1:])

            if metadata is not None:
                questions: List[str] = []
                query_ids: List[str] = []
                question_tokens: List[List[str]] = []
                gold_program_tokens: List[List[str]] = []
                for i in range(0, len(metadata)):
                    questions.append(metadata[i]["question"])
                    query_ids.append(metadata[i]["query_id"])
                    question_tokens.append(metadata[i]["utterance_tokens"])
                    if "target_tokens" in metadata[i]:
                        gold_program_tokens.append(metadata[i]["target_tokens"])

                output_dict["question"] = questions
                output_dict["query_id"] = query_ids
                output_dict["question_tokens"] = question_tokens
                output_dict["gold_program_tokens"] = gold_program_tokens
                output_dict["exact_match"] = exact_matches

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides `Model.make_output_human_readable`, which gets called after `Model.forward`, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            predicted_tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def exact_match_sequences(self, predicted_indices, gold_indices):
        exact_matches = []
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        if not isinstance(gold_indices, numpy.ndarray):
            gold_indices = gold_indices.detach().cpu().numpy()
        for p_indices, g_indices in zip(predicted_indices, gold_indices):
            p_indices = list(p_indices)
            g_indices = list(g_indices)
            if self._end_index in p_indices:
                p_indices = p_indices[: p_indices.index(self._end_index)]
            if self._end_index in g_indices:
                g_indices = g_indices[: g_indices.index(self._end_index)]
            if len(p_indices) == len(g_indices):
                correct = int(all([x == y for x, y in zip(p_indices, g_indices)]))
            else:
                correct = 0
            self._exact_match(value=correct)
            exact_matches.append(correct)
        return exact_matches

    def get_sequence_indices_and_tokens(self, batch_sequences):
        if not isinstance(batch_sequences, numpy.ndarray):
            batch_sequences = batch_sequences.detach().cpu().numpy()
            # Shape of batch_sequences: (batch_size, sequence_length)
            assert len(batch_sequences.shape) == 2
            batch_sequences = batch_sequences.tolist()
        batch_indices = []
        batch_tokens = []
        longest_sequence = -1
        for i in range(0, len(batch_sequences)):
            indices = batch_sequences[i]
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            longest_sequence = len(indices) if len(indices) > longest_sequence else longest_sequence
            batch_indices.append(indices)
            batch_tokens.append(tokens)
        return batch_indices, batch_tokens, longest_sequence

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        embedded_input = self._dropout(embedded_input)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        encoder_outputs = self._dropout(encoder_outputs)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"], state["source_mask"], self._encoder.is_bidirectional()
        )
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )
        state["coverage_context"] = state["encoder_outputs"].new_zeros(batch_size, state["encoder_outputs"].shape[1])
        return state

    def _forward_loop(
        self, state: Dict[str, torch.Tensor], target_tokens: TextFieldTensors = None,
        attention_supervision: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.

        Additional
        ----------
        attention_supervision: `(batch_size, num_target_tokens, num_input_tokens)`
        For each target token, multi-one-hot for inputs to attend to
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]
        target_mask = None
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]["tokens"]
            target_mask = util.get_text_field_mask(target_tokens)

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps


        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        step_attn_weights: List[torch.Tensor] = []
        attention_supervision_loss = 0.0
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            t_mask = None
            if target_mask is not None:
                t_mask = target_mask[:, timestep]   # Shape: (batch_size)

            attn_supvervision = None
            if attention_supervision is not None and timestep != num_decoding_steps - 1:    # No sup for decoding @END@
                attn_supvervision = attention_supervision[:, timestep, :]   # Shape: (batch_size, num_input_tokens)

            # shape: (batch_size, num_classes)
            input_weights, output_projections, state = self._prepare_output_projections(input_choices, state,
                                                                                        attn_supvervision,
                                                                                        t_mask)
            step_attn_weights.append(input_weights.unsqueeze(1))

            if self.training and "attention_loss" in state:
                attention_supervision_loss += state["attention_loss"]
                state.pop("attention_loss")

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        # shape: (batch_size, num_decoding_steps, max_input_sequence_length)
        attention_input_weights = torch.cat(step_attn_weights, 1)

        output_dict = {"predictions": predictions, "attention_input_weights": attention_input_weights}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)

            coverage_loss = self._get_coverage_loss(attention_input_weights, source_mask, target_mask)
            self._coverage_loss(coverage_loss.detach().cpu().item())

            attention_supervision_loss = attention_supervision_loss / (num_decoding_steps - 1)
            output_dict["loss"] = loss + attention_supervision_loss
            output_dict["s2s_loss"] = loss
            output_dict["cov_loss"] = coverage_loss
            if self.training:
                output_dict["attn_loss"] = attention_supervision_loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor],
        attention_supervision: torch.Tensor = None, t_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, max_input_sequence_length)
        coverage_context = state["coverage_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        # shape: (group_size, encoder_output_dim)
        attended_input, input_weights, attn_loss = self._prepare_attended_input(
            decoder_hidden, encoder_outputs, source_mask, coverage_context,
            attention_supervision, t_mask,
        )
        if self.training:
            state["attention_loss"] = attn_loss

        # shape: (group_size, decoder_output_dim + target_embedding_dim)
        decoder_input = torch.cat((attended_input, embedded_input), -1)
        decoder_input = self._dropout(decoder_input)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["coverage_context"] = torch.add(coverage_context, input_weights)

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(self._dropout(decoder_hidden))

        return input_weights, output_projections, state

    def _prepare_attended_input(
        self,
        decoder_hidden_state: torch.Tensor = None,
        encoder_outputs: torch.Tensor = None,
        encoder_outputs_mask: torch.BoolTensor = None,
        coverage_context: torch.Tensor = None,
        attention_supervision: torch.Tensor = None,
        t_mask: torch.Tensor = None,
    ):
        """Apply attention over encoder outputs and decoder state.

        attention_supervision: (batch_size, num_inputs) -- For this step, inputs to attend to
        t_mask: (batch_size,) -- Mask for this decoding time-step
        """
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, encoder_outputs, coverage_context, encoder_outputs_mask)

        if attention_supervision is None:
            attn_loss = 0
        else:
            # Shape: (batch_size,) -- should be 0s if superivision is to attend to no tokens
            attn_sup_mask = (torch.sum(attention_supervision, dim=1) > 0).float()
            # Shape: (batch_size, num_inputs)
            attended_probabilities = input_weights * attention_supervision * encoder_outputs_mask * \
                                     t_mask.unsqueeze(1).float() * attn_sup_mask.unsqueeze(1).float()
            # Maximize the sum of probabities
            attended_probabilities = torch.sum(attended_probabilities, dim=1)
            log_sum_probs = torch.log(attended_probabilities + 1e-15)
            attn_loss = -1.0 * torch.mean(log_sum_probs)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input, input_weights, attn_loss

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @staticmethod
    def _get_coverage_loss(attn_weights: torch.Tensor,
                           source_mask: torch.LongTensor,
                           target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the coverage loss: the minimal between the weight assigned to each input token in the current step,
        and the total weight assigned to it on previous steps, summed across all input tokens and steps.
        """
        # shape: (batch_size, max_decoding_steps, max_input_seq_length
        attn_weights = attn_weights.float()
        # shape: (batch_size, max_decoding_steps)
        target_mask = target_mask.float()
        # shape: (batch_size, max_input_seq_length)
        source_mask = source_mask.float()

        batch_size = attn_weights.size(0)
        max_decoding_steps = attn_weights.size(1)
        max_input_seq_length = attn_weights.size(2)

        # shape : (batch_size,)
        actual_steps_per_batch = target_mask.sum(dim=1)

        all_contexts = attn_weights.cumsum(1)
        # we lose the first attention vector (as there's no context to compare with),
        # and the last context vector (as there's attention vector to compare with)
        loss_per_entry = torch.min(attn_weights[:, 1:, :], all_contexts[:, :-1, :])

        loss_per_entry = loss_per_entry * source_mask.view(batch_size, 1, max_input_seq_length).expand_as(
            loss_per_entry)
        loss_per_entry = loss_per_entry * target_mask[:, 1:-1].view(batch_size, max_decoding_steps - 1, 1).expand_as(
            loss_per_entry)

        # average across all timesteps and batches (between 0 to 1)
        per_batch_loss = loss_per_entry.sum(dim=(1, 2)) / (actual_steps_per_batch + 1e-13)
        return per_batch_loss.sum() / batch_size

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        all_metrics.update({'coverage_loss': self._coverage_loss.get_metric(reset=reset)})
        if not self.training:
            exact_match = self._exact_match.get_metric(reset=reset)
            all_metrics.update({'exact_match': exact_match})
            if self._bleu:
                all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics