from typing import Any, Dict, List, Optional
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1
from pytorch_pretrained_bert import BertModel, BertTokenizer
import pickle

from drop_bert.nhelpers import tokenlist_to_passage, beam_search, evaluate_postfix

logger = logging.getLogger(__name__)

@Model.register("nabertT")
class NumericallyAugmentedBERTT(Model):
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
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None,
                 number_rep: str = 'first',
                 special_numbers : List[int] = None) -> None:
        super().__init__(vocab, regularizer)

        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "arithmetic", "counting"]
        else:
            self.answering_abilities = answering_abilities
        self.number_rep = number_rep
        
        self.BERT = BertModel.from_pretrained(bert_pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        bert_dim = self.BERT.pooler.dense.out_features
        
        self.dropout = dropout_prob

        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._number_weights_predictor = torch.nn.Linear(bert_dim, 1)
            
        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = \
                self.ff(2 * bert_dim, bert_dim, len(self.answering_abilities))

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = torch.nn.Linear(bert_dim, 1)
            self._passage_span_end_predictor = torch.nn.Linear(bert_dim, 1)

        if "question_span_extraction" in self.answering_abilities:
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = \
                self.ff(2 * bert_dim, bert_dim, 1)
            self._question_span_end_predictor = \
                self.ff(2 * bert_dim, bert_dim, 1)

        if "arithmetic" in self.answering_abilities:
            self._arithmetic_index = self.answering_abilities.index("arithmetic")
            self.special_numbers = special_numbers
            self.num_special_numbers = len(self.special_numbers)
            self.special_embedding = torch.nn.Embedding(self.num_special_numbers, bert_dim)
            self.num_arithmetic_templates = 5
            self.num_template_slots = 3
            self._arithmetic_template_predictor = self.ff(2 * bert_dim, bert_dim, self.num_arithmetic_templates)
            self._arithmetic_template_slot_predictor = \
                torch.nn.Linear(2 * bert_dim, self.num_arithmetic_templates * self.num_template_slots)
            
            self._arithmetic_passage_weight_predictor = torch.nn.Linear(bert_dim, 1)
            self._arithmetic_question_weight_predictor = torch.nn.Linear(bert_dim, 1)
            
            self.templates = [lambda x,y,z: (x + y) * z,
                              lambda x,y,z: (x - y) * z,
                              lambda x,y,z: (x + y) / z,
                              lambda x,y,z: (x - y) / z,
                              lambda x,y,z: x * y / z]

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = \
                self.ff(bert_dim, bert_dim, max_count + 1) 

        self._drop_metrics = DropEmAndF1()
        initializer(self)

    def summary_vector(self, encoding, mask, in_type = "passage"):
        if in_type == "passage":
            # Shape: (batch_size, seqlen)
            alpha = self._passage_weights_predictor(encoding).squeeze()
        elif in_type == "question":
            # Shape: (batch_size, seqlen)
            alpha = self._question_weights_predictor(encoding).squeeze()
        elif in_type == "arithmetic_passage":
            # Shape: (batch_size, seqlen)
            alpha = self._arithmetic_passage_weight_predictor(encoding).squeeze()
        elif in_type == "arithmetic_question":
            # Shape: (batch_size, seqlen)
            alpha = self._arithmetic_question_weight_predictor(encoding).squeeze()
        else:
            # Shape: (batch_size, #num of numbers, seqlen)
            alpha = torch.zeros(encoding.shape[:-1], device=encoding.device)
            if self.number_rep == 'attention':
                alpha = self._number_weights_predictor(encoding).squeeze()     
        # Shape: (batch_size, seqlen) 
        # (batch_size, #num of numbers, seqlen) for numbers
        alpha = masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        # (batch_size, #num of numbers, out) for numbers
        h = util.weighted_sum(encoding, alpha)
        return h
    
    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(self.dropout),
                                   torch.nn.Linear(hidden_dim, output_dim))

    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                mask_indices: torch.LongTensor,
                num_spans: torch.LongTensor = None,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_expressions: torch.LongTensor = None,
                answer_as_expressions_extra: torch.LongTensor = None,
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
        question_vector = self.summary_vector(question_out, question_mask, "question")
        
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
            count_number_log_probs, best_count_number = self._count_module(passage_vector)

        if "passage_span_extraction" in self.answering_abilities:
            passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span = \
                self._passage_span_module(passage_out, passage_mask)

        if "question_span_extraction" in self.answering_abilities:
            question_span_start_log_probs, question_span_end_log_probs, best_question_span = \
                self._question_span_module(passage_vector, question_out, question_mask)
            
        if "arithmetic" in self.answering_abilities:
            arithmetic_passage_vector = self.summary_vector(passage_out, passage_mask, "arithmetic_passage")
            arithmetic_question_vector = self.summary_vector(question_out, question_mask, "arithmetic_question")
            
            arithmetic_template_logits = \
                self._arithmetic_template_predictor(torch.cat([arithmetic_passage_vector, arithmetic_question_vector], -1))
            arithmetic_template_log_probs = arithmetic_template_logits.log_softmax(-1)
            arithmetic_best_templates = arithmetic_template_log_probs.argmax(-1)
            
            number_mask = (number_indices[:,:,0].long() != -1).long()
            
            arithmetic_template_slot_log_probs, arithmetic_best_template_slots, number_mask = \
                self._arithmetic_module(arithmetic_passage_vector, passage_out, number_indices, number_mask)

            
        output_dict = {}
        del passage_out, question_out
        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None \
                or answer_as_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    log_marginal_likelihood_for_passage_span = \
                        self._passage_span_log_likelihood(answer_as_passage_spans,
                                                          passage_span_start_log_probs,
                                                          passage_span_end_log_probs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    log_marginal_likelihood_for_question_span = \
                        self._question_span_log_likelihood(answer_as_question_spans,
                                                           question_span_start_log_probs,
                                                           question_span_end_log_probs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)

                elif answering_ability == "arithmetic":
                    log_marginal_likelihood_for_arithmetic = \
                        self._arithmetic_log_likelihood(answer_as_expressions,
                                                        arithmetic_template_slot_log_probs, 
                                                        arithmetic_template_log_probs)                                  
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_arithmetic)

                elif answering_ability == "counting":
                    log_marginal_likelihood_for_count = \
                        self._count_log_likelihood(answer_as_counts, 
                                                   count_number_log_probs)
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
        with torch.no_grad():
            # Compute the metrics and add the tokenized input to the output.
            if metadata is not None:
                output_dict["question_id"] = []
                output_dict["answer"] = []
                question_tokens = []
                passage_tokens = []
                for i in range(batch_size):
                    if len(self.answering_abilities) > 1:
                        predicted_ability_str = self.answering_abilities[best_answer_ability[i]]
                    else:
                        predicted_ability_str = self.answering_abilities[0]
                    answer_json: Dict[str, Any] = {}

                    # We did not consider multi-mention answers here
                    if predicted_ability_str == "passage_span_extraction":
                        answer_json["answer_type"] = "passage_span"
                        answer_json["value"], answer_json["spans"] = \
                            self._span_prediction(question_passage_tokens[i], best_passage_span[i])
                    elif predicted_ability_str == "question_span_extraction":
                        answer_json["answer_type"] = "question_span"
                        answer_json["value"], answer_json["spans"] = \
                            self._span_prediction(question_passage_tokens[i], best_question_span[i])
                    elif predicted_ability_str == "arithmetic":  
                        answer_json["answer_type"] = "arithmetic"
                        original_numbers = metadata[i]['original_numbers']
                        answer_json["value"], answer_json["indices"], answer_json["numbers"] = \
                            self._arithmetic_prediction(original_numbers, 
                                                             arithmetic_best_templates[i],
                                                             arithmetic_best_template_slots[i])
                        answer_json['template'] = arithmetic_best_templates[i].item()
                    elif predicted_ability_str == "counting":
                        answer_json["answer_type"] = "count"
                        answer_json["value"], answer_json["count"] = \
                            self._count_prediction(best_count_number[i])
                    else:
                        raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                    output_dict["question_id"].append(metadata[i]["question_id"])
                    output_dict["answer"].append(answer_json)
                    answer_annotations = metadata[i].get('answer_annotations', [])
                    if answer_annotations:
                        self._drop_metrics(answer_json["value"], answer_annotations)

        return output_dict
    
    
    def _passage_span_module(self, passage_out, passage_mask):
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

        # Shape: (batch_size, 2)
        best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
        return passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span
    
    
    def _passage_span_log_likelihood(self,
                                     answer_as_passage_spans,
                                     passage_span_start_log_probs,
                                     passage_span_end_log_probs):
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
        return log_marginal_likelihood_for_passage_span
    
    
    def _span_prediction(self, question_passage_tokens, best_span):
        (predicted_start, predicted_end)  = tuple(best_span.detach().cpu().numpy())
        answer_tokens = question_passage_tokens[predicted_start:predicted_end + 1].detach().cpu().numpy()
        token_lst = self.tokenizer.convert_ids_to_tokens(answer_tokens)
        predicted_answer = tokenlist_to_passage(token_lst)
        return predicted_answer, [(predicted_start, predicted_end)]

    
    def _question_span_module(self, passage_vector, question_out, question_mask):
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

        # Shape: (batch_size, 2)
        best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)
        return question_span_start_log_probs, question_span_end_log_probs, best_question_span
    
    
    def _question_span_log_likelihood(self, 
                                      answer_as_question_spans, 
                                      question_span_start_log_probs, 
                                      question_span_end_log_probs):
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
        return log_marginal_likelihood_for_question_span
    
    
    def _count_module(self, passage_vector):
        # Shape: (batch_size, 10)
        count_number_logits = self._count_number_predictor(passage_vector)
        count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_count_number = torch.argmax(count_number_log_probs, -1)
        return count_number_log_probs, best_count_number
    

    def _count_log_likelihood(self, answer_as_counts, count_number_log_probs):
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
        return log_marginal_likelihood_for_count
    
    
    def _count_prediction(self, best_count_number):
        predicted_count = best_count_number.detach().cpu().numpy()
        predicted_answer = str(predicted_count)
        return predicted_answer, predicted_count
    
    
    def _arithmetic_module(self, arithmetic_passage_vector, passage_out, number_indices, number_mask):
        if self.number_rep in ['average', 'attention']:
            
            # Shape: (batch_size, # of numbers, # of pieces) 
            number_indices = util.replace_masked_values(number_indices, number_indices != -1, 0).long()
            batch_size = number_indices.shape[0]
            num_numbers = number_indices.shape[1]
            seqlen = passage_out.shape[1]

            # Shape : (batch_size, # of numbers, seqlen)
            mask = torch.zeros((batch_size, num_numbers, seqlen), device=number_indices.device).long().scatter(
                        2, 
                        number_indices, 
                        torch.ones(number_indices.shape, device=number_indices.device).long())
            mask[:,:,0] = 0

            # Shape : (batch_size, # of numbers, seqlen, bert_dim)
            epassage_out = passage_out.unsqueeze(1).repeat(1,num_numbers,1,1)

            # Shape : (batch_size, # of numbers, bert_dim)
            encoded_numbers = self.summary_vector(epassage_out, mask, "numbers")
        else:
            number_indices = number_indices[:,:,0].long()
            clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
            encoded_numbers = torch.gather(
                    passage_out,
                    1,
                    clamped_number_indices.unsqueeze(-1).expand(-1, -1, passage_out.size(-1)))
            
        if self.num_special_numbers > 0:
            special_numbers = self.special_embedding(torch.arange(self.num_special_numbers, device=number_indices.device))
            special_numbers = special_numbers.expand(number_indices.shape[0],-1,-1)
            encoded_numbers = torch.cat([special_numbers, encoded_numbers], 1)
            
            mask = torch.ones((number_indices.shape[0],self.num_special_numbers), device=number_indices.device).long()
            number_mask = torch.cat([mask, number_mask], -1)
            
        # Shape: (batch_size, # of numbers, 2*bert_dim)
        encoded_numbers = torch.cat(
                [encoded_numbers, arithmetic_passage_vector.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)], -1)
        
        # Shape: (batch_size, #templates, #slots, #numbers)
        arithmetic_template_slot_logits = self._arithmetic_template_slot_predictor(encoded_numbers).transpose(1,2)
        arithmetic_template_slot_log_probs = util.masked_log_softmax(arithmetic_template_slot_logits, number_mask)
        arithmetic_template_slot_log_probs = arithmetic_template_slot_log_probs.reshape(number_mask.shape[0],
                                                                                        self.num_arithmetic_templates, 
                                                                                        self.num_template_slots, 
                                                                                        number_mask.shape[-1])
        # Shape: (batch_size, #templates, #slots)
        arithmetic_best_template_slots = arithmetic_template_slot_log_probs.argmax(-1)
        return arithmetic_template_slot_log_probs, arithmetic_best_template_slots, number_mask
    
    
    def _arithmetic_log_likelihood(self,
                                    answer_as_expressions,
                                    arithmetic_template_slot_log_probs, 
                                    arithmetic_template_log_probs):
        # answer_as_expressions : (batch, #templates, #expressions, #slots)
        # arithmetic_template_slot_log_probs : (batch, #templates, #slots, #numbers)
        # arithmetic_template_log_probs : (batch, #templates)
        
        # shape : (batch, #templates, #slots, #expressions)
        gold_templates = answer_as_expressions.transpose(2,3).long()
        
        # mask for invalid/padded expressions
        gold_templates_mask = (gold_templates[:,:,:,:] != -1).long()
        clamped_gold_templates = \
            util.replace_masked_values(gold_templates, gold_templates_mask, 0)
        
        # shape : (batch, #templates, #slots, #expressions)
        log_likelihood_per_slot = \
            torch.gather(arithmetic_template_slot_log_probs, -1, clamped_gold_templates)
        
        # shape : (batch, #templates, #expressions)
        log_likelihood_per_expression = log_likelihood_per_slot.sum(2)
        # mask out padded expressions
        log_likelihood_per_expression = util.replace_masked_values(log_likelihood_per_expression, 
                                                                   gold_templates_mask[:,:,0,:], 
                                                                   -1e7)
        # shape : (batch, #templates)
        log_likelihood_per_template = util.logsumexp(log_likelihood_per_expression)
        log_joint_likelihood_for_arithmetic = log_likelihood_per_template + arithmetic_template_log_probs
        
        # Shape: (batch_size, )
        log_marginal_likelihood_for_arithmetic = util.logsumexp(log_joint_likelihood_for_arithmetic)
        return log_marginal_likelihood_for_arithmetic
    
    
    def _arithmetic_prediction(self, original_numbers, best_template, best_template_slots):
        original_numbers = self.special_numbers + original_numbers
        indices = best_template_slots[best_template].cpu().numpy().tolist()
        numbers = [original_numbers[indices[i]] for i in range(self.num_template_slots)]
        try:
            predicted_answer = str(float(self.templates[best_template](*numbers)))
        except:
            predicted_answer = "0.0"
        return predicted_answer, indices, numbers 
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}