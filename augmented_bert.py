from typing import Any, Dict, List, Optional
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1
from pytorch_pretrained_bert import BertModel, BertTokenizer


from drop_nmn.nhelpers import tokenlist_to_passage

logger = logging.getLogger(__name__)

@Model.register("nabert")
class NumericallyAugmentedBERT(Model):
    """
    This class augments BERT with some rudimentary numerical reasoning abilities. This is based on
    NAQANet, as published in the original DROP paper. The code is based on the AllenNLP 
    implementation of NAQANet
    """
    def __init__(self, 
                 vocab: Vocabulary, 
                 bert_pretrained_model: str, 
                 dropout_prob: float = 0.1, 
                 max_count: int = 10,
                 max_spans: int = 10,
                 multi_span: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None) -> None:
        super().__init__(vocab, regularizer)

        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting"]
        else:
            self.answering_abilities = answering_abilities
        self.multi_span = multi_span
        
        self.BERT = BertModel.from_pretrained(bert_pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        bert_dim = self.BERT.pooler.dense.out_features
        
        self.dropout = dropout_prob

        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(bert_dim, 1)
        
        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = \
                self.ff(2 * bert_dim, bert_dim, len(self.answering_abilities))

        if "passage_span_extraction" in self.answering_abilities:
            if self.multi_span:
                self._num_passage_span_predictor = \
                    self.ff(2 * bert_dim, bert_dim, max_spans + 1)
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = torch.nn.Linear(bert_dim, 1)
            self._passage_span_end_predictor = torch.nn.Linear(bert_dim, 1)

        if "question_span_extraction" in self.answering_abilities:
            if self.multi_span:
                self._num_question_span_predictor = \
                    self.ff(2 * bert_dim, bert_dim, max_spans + 1)
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = \
                self.ff(2 * bert_dim, bert_dim, 1)
            self._question_span_end_predictor = \
                self.ff(2 * bert_dim, bert_dim, 1)

        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = \
                self.ff(2 * bert_dim, bert_dim, 3)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = \
                self.ff(bert_dim, bert_dim, max_count + 1) 

        self._drop_metrics = DropEmAndF1()
        self.device = torch.cuda.current_device()  # torch.device('cpu')
        initializer(self)

    def summary_vector(self, encoding, mask, passage = True):
        if passage:
            # Shape: (batch_size, seqlen)
            alpha = self._passage_weights_predictor(encoding).squeeze()
        else:
            # Shape: (batch_size, seqlen)
            alpha = self._question_weights_predictor(encoding).squeeze()
        # Shape: (batch_size, seqlen)
        alpha = masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        h = util.weighted_sum(encoding, alpha)
        return h
    
    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(self.dropout),
                                   torch.nn.Linear(hidden_dim, output_dim))
    
    def get_best_span(self, span_start_logits, span_end_logits, K = 1):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
        valid_span_log_probs = span_log_probs + span_log_mask
        valid_span_probs = valid_span_log_probs.exp()

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        starts = []
        ends = []
        best_spans = valid_span_probs.log().view(batch_size, -1).argmax(-1)
        start_ind = best_spans // passage_length
        end_ind = best_spans % passage_length
        starts.append(start_ind)
        ends.append(end_ind)

        for _ in range(K - 1):
            for b in range(batch_size):
                valid_span_probs[b,start_ind[b].item():end_ind[b].item() + 1] = 0
                valid_span_probs[b,:start_ind[b].item(),start_ind[b].item():] = 0
            best_spans = valid_span_probs.log().view(batch_size, -1).argmax(-1)
            start_ind = best_spans // passage_length
            end_ind = best_spans % passage_length
            starts.append(start_ind)
            ends.append(end_ind)
        starts = torch.stack(starts, dim=-1)
        ends = torch.stack(ends, dim=-1)

        return torch.stack([starts, ends], dim=-1)

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                mask_indices: torch.LongTensor,
                num_spans: torch.LongTensor = None,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        
        # Shape: (batch_size, seqlen)
        question_passage_tokens = question_passage["tokens"]
        # Shape: (batch_size, seqlen)
        pad_mask = question_passage["mask"] 
        # Shape: (batch_size, seqlen)
        seqlen_ids = question_passage["tokens-type-ids"]
        
        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]
                
        # Shape: (batch_size, 3)
        mask = mask_indices.squeeze(-1)
        # Shape: (batch_size, seqlen)
        cls_sep_mask = \
            torch.ones(pad_mask.shape, device=pad_mask.device).long().scatter(1, mask, torch.zeros(mask.shape, device=mask.device).long())
        # Shape: (batch_size, seqlen)
        passage_mask = seqlen_ids * pad_mask * cls_sep_mask
        # Shape: (batch_size, seqlen)
        question_mask = (1 - seqlen_ids) * pad_mask * cls_sep_mask
        
        # Shape: (batch_size, seqlen, bert_dim)
        bert_out, _ = self.BERT(question_passage_tokens, seqlen_ids, pad_mask, output_all_encoded_layers=False)

        # Shape: (batch_size, qlen, bert_dim)
        question_end = max(mask[:,1])
        question_out = bert_out[:,:question_end]
        # Shape: (batch_size, qlen)
        question_mask = question_mask[:,:question_end]
        # Shape: (batch_size, out)
        question_vector = self.summary_vector(question_out, question_mask, False)

        passage_out = bert_out
        del bert_out
        # Shape: (batch_size, bert_dim)
        passage_vector = self.summary_vector(passage_out, passage_mask)

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        if "counting" in self.answering_abilities:
            # Shape: (batch_size, 10)
            count_number_logits = self._count_number_predictor(passage_vector)
            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
            # Info about the best count number prediction
            # Shape: (batch_size,)
            best_count_number = torch.argmax(count_number_log_probs, -1)

        if "passage_span_extraction" in self.answering_abilities:
            if self.multi_span:
                # Shape : (batch_size, max_#_spans)
                num_passage_span_logits = \
                    self._num_passage_span_predictor(torch.cat([passage_vector, question_vector], -1))
                num_passage_span_log_probs = torch.nn.functional.log_softmax(num_passage_span_logits, -1)
            
            # Shape: (batch_size, passage_length)
            passage_span_start_logits = self._passage_span_start_predictor(passage_out).squeeze(-1)

            # Shape: (batch_size, passage_length)
            passage_span_end_logits = self._passage_span_end_predictor(passage_out).squeeze(-1)

            # Shape: (batch_size, passage_length)
            passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
            passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

            # Info about the best passage span prediction
            passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
            passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)
            
            max_best_num_passage_span = 1
            if self.multi_span:
                # Shape: (batch_size, max_best_num, 2)
                best_num_passage_span = torch.argmax(num_passage_span_log_probs, -1)
                max_best_num_passage_span = torch.max(best_num_passage_span)
            
            # Shape: (batch_size, 2)
            best_passage_span = \
                self.get_best_span(passage_span_start_logits, passage_span_end_logits, max_best_num_passage_span)

        if "question_span_extraction" in self.answering_abilities:
            # Shape : (batch_size, max_#_spans)
            if self.multi_span:
                num_question_span_logits = \
                    self._num_question_span_predictor(torch.cat([passage_vector, question_vector], -1))
                num_question_span_log_probs = torch.nn.functional.log_softmax(num_question_span_logits, -1)
            
            # Shape: (batch_size, question_length)
            encoded_question_for_span_prediction = \
                torch.cat([question_out,
                           passage_vector.unsqueeze(1).repeat(1, question_out.size(1), 1)], -1)
            question_span_start_logits = \
                self._question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)
            # Shape: (batch_size, question_length)
            question_span_end_logits = \
                self._question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)
            question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, question_mask)
            question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, question_mask)

            # Info about the best question span prediction
            question_span_start_logits = \
                util.replace_masked_values(question_span_start_logits, question_mask, -1e7)
            question_span_end_logits = \
                util.replace_masked_values(question_span_end_logits, question_mask, -1e7)
            
            max_best_num_question_span = 1
            if self.multi_span:
                # Shape: (batch_size, max_best_num, 2)
                best_num_question_span = torch.argmax(num_question_span_log_probs, -1)
                max_best_num_question_span = torch.max(best_num_question_span)
            
            # Shape: (batch_size, 2)
            best_question_span = \
                self.get_best_span(question_span_start_logits, question_span_end_logits, max_best_num_question_span)

        if "addition_subtraction" in self.answering_abilities:
            # Shape: (batch_size, # of numbers in the passage)
            number_indices = number_indices[:,:,0].long()
            number_mask = (number_indices != -1).long()
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            
            # num_mask_indices: (batch_size, # of numbers, # of pieces)
            # number_indices: (batch_size, # of numbers)
            # passage_out: (batch_size, seqlen, encoding_dim)
            
            # passage_out.dot(mask): (batch_size, # of numbers, bert_dim)
            # sum(mask): (batch_size, # of numbers)
            # paassage_out.dot(mask) / sum(mask): (batch_size, # of numbers, bert_dim)

            # Shape: (batch_size, # of numbers in the passage, encoding_dim)
            encoded_numbers = torch.gather(
                    passage_out,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(-1, -1, passage_out.size(-1)))
            # Shape: (batch_size, # of numbers in the passage, 2 * bert_dim)
            encoded_numbers = torch.cat(
                    [encoded_numbers, passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)

            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_logits = self._number_sign_predictor(encoded_numbers)
            number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

            # Shape: (batch_size, # of numbers in passage).
            best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
            # For padding numbers, the best sign masked as 0 (not included).
            best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)

        output_dict = {}

        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None \
                or answer_as_add_sub_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    log_likelihood_for_num_passage_spans = 0
                    if self.multi_span:
                        # Shape: (batch_size, )
                        log_likelihood_for_num_passage_spans = \
                            torch.gather(num_passage_span_log_probs, 1, num_spans.unsqueeze(-1)).squeeze()
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()
                    clamped_gold_passage_span_starts = \
                        util.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = \
                        util.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = \
                        torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
                    log_likelihood_for_passage_span_ends = \
                        torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = \
                        log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = \
                        util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
                    log_marginal_likelihood_for_passage_span += log_likelihood_for_num_passage_spans
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    log_likelihood_for_num_question_spans = 0
                    if self.multi_span:
                        # Shape: (batch_size, )
                        log_likelihood_for_num_question_spans = \
                            torch.gather(num_question_span_log_probs, 1, num_spans.unsqueeze(-1)).squeeze()
                    
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).long()
                    clamped_gold_question_span_starts = \
                        util.replace_masked_values(gold_question_span_starts, gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = \
                        util.replace_masked_values(gold_question_span_ends, gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = \
                        torch.gather(question_span_start_log_probs, 1, clamped_gold_question_span_starts)
                    log_likelihood_for_question_span_ends = \
                        torch.gather(question_span_end_log_probs, 1, clamped_gold_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = \
                        log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_question_spans = \
                        util.replace_masked_values(log_likelihood_for_question_spans,
                                                   gold_question_span_mask,
                                                   -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_question_span = \
                        util.logsumexp(log_likelihood_for_question_spans)
                    log_marginal_likelihood_for_question_span += log_likelihood_for_num_question_spans
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)

                elif answering_ability == "addition_subtraction":
                    # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
                    # Shape: (batch_size, # of combinations)
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum(-1) > 0).float()
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    gold_add_sub_signs = answer_as_add_sub_expressions.transpose(1, 2)
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                    # the log likelihood of the masked positions should be 0
                    # so that it will not affect the joint probability
                    log_likelihood_for_number_signs = \
                        util.replace_masked_values(log_likelihood_for_number_signs, number_mask.unsqueeze(-1), 0)
                    # Shape: (batch_size, # of combinations)
                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                    # For those padded combinations, we set their log probabilities to be very small negative value
                    log_likelihood_for_add_subs = \
                        util.replace_masked_values(log_likelihood_for_add_subs, gold_add_sub_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = (answer_as_counts != -1).long()
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = \
                        util.replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]

            output_dict["loss"] = - marginal_log_likelihood.mean()

        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])

                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[i].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]
                answer_json: Dict[str, Any] = {}

                # We did not consider multi-mention answers here
                if predicted_ability_str == "passage_span_extraction":
                    answer_json["answer_type"] = "passage_span"
                    num_spans = best_num_passage_span[i]
                    predicted_answers = []
                    predicted_spans = []
                    for j in range(num_spans):
                        (predicted_start, predicted_end)  = tuple(best_passage_span[i,j].detach().cpu().numpy())
                        answer_tokens = question_passage_tokens[i][predicted_start:predicted_end + 1].detach().cpu().numpy()
                        token_lst = self.tokenizer.convert_ids_to_tokens(answer_tokens)
                        predicted_answer = tokenlist_to_passage(token_lst)
                        predicted_answers.append(predicted_answer)
                        predicted_spans.append((predicted_start, predicted_end))
                    answer_json["value"] = predicted_answers
                    answer_json["spans"] = predicted_spans
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    num_spans = best_num_question_span[i]
                    predicted_answers = []
                    predicted_spans = []
                    for j in range(num_spans):
                        (predicted_start, predicted_end)  = tuple(best_question_span[i,j].detach().cpu().numpy())
                        answer_tokens = question_passage_tokens[i][predicted_start:predicted_end + 1].detach().cpu().numpy()
                        token_lst = self.tokenizer.convert_ids_to_tokens(answer_tokens)
                        predicted_answer = tokenlist_to_passage(token_lst)
                        predicted_answers.append(predicted_answer)
                        predicted_spans.append((predicted_start, predicted_end))
                    answer_json["value"] = predicted_answers
                    answer_json["spans"] = predicted_spans
                elif predicted_ability_str == "addition_subtraction":  # plus_minus combination answer
                    answer_json["answer_type"] = "arithmetic"
                    original_numbers = metadata[i]['original_numbers']
                    sign_remap = {0: 0, 1: 1, 2: -1}
                    predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                    result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                    predicted_answer = str(result)
                    answer_json['numbers'] = []
                    for value, sign in zip(original_numbers, predicted_signs):
                        answer_json['numbers'].append({'value': value, 'sign': sign})
                    if number_indices[i][-1] == -1:
                        # There is a dummy 0 number at position -1 added in some cases; we are
                        # removing that here.
                        answer_json["numbers"].pop()
                    answer_json["value"] = result
                elif predicted_ability_str == "counting":
                    answer_json["answer_type"] = "count"
                    predicted_count = best_count_number[i].detach().cpu().numpy()
                    predicted_answer = str(predicted_count)
                    answer_json["count"] = predicted_count
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations)

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}