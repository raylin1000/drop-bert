from namedtensor import ntorch
import torch
from drop_bert.activations import ReLU, Identity
import drop_bert.nhelpers as nhelpers
from allennlp.models.model import Model
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1


"""
A feed-forward neural network.
"""
class FeedForward(ntorch.nn.Module):
    """
    This ``Module`` is a feed-forward neural network, just a sequence of ``Linear`` layers with
    activation functions in between.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    num_layers : ``int``
        The number of ``Linear`` layers to apply to the input.
    hidden_dims : ``Union[int, Sequence[int]]``
        The output dimension of each of the ``Linear`` layers.  If this is a single ``int``, we use
        it for all ``Linear`` layers.  If it is a ``Sequence[int]``, ``len(hidden_dims)`` must be
        ``num_layers``.
    activations : ``Union[Callable, Sequence[Callable]]``
        The activation function to use after each ``Linear`` layer.  If this is a single function,
        we use it after all ``Linear`` layers.  If it is a ``Sequence[Callable]``,
        ``len(activations)`` must be ``num_layers``.
    dropout : ``Union[float, Sequence[float]]``, optional
        If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
        versus ``Sequence[float]`` is the same as with other parameters.
    """
    def __init__(self, input_dim, num_layers, hidden_dims, activations,
                 dropout = 0.0, out_name = "out"):
        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise ConfigurationError("len(hidden_dims) (%d) != num_layers (%d)" %
                                     (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise ConfigurationError("len(activations) (%d) != num_layers (%d)" %
                                     (len(activations), num_layers))
        if len(dropout) != num_layers:
            raise ConfigurationError("len(dropout) (%d) != num_layers (%d)" %
                                     (len(dropout), num_layers))
        self._activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(ntorch.nn.Linear(layer_input_dim, layer_output_dim).spec(out_name))
        self._linear_layers = ntorch.nn.ModuleList(linear_layers)
        dropout_layers = [ntorch.nn.Dropout(p=value) for value in dropout]
        self._dropout = ntorch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs):
        # pylint: disable=arguments-differ
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output

@Model.register("aug_bert")
class AugmentedBERT(Model):
    """Numerically Augemented BERT for DROP"""
    def __init__(self, vocab, BERT, BERT_tokenizer, answering_abilities = None, dropout = 0., regularizer = None):
        super().__init__(vocab, regularizer)
        
        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting"]
        else:
            self.answering_abilities = answering_abilities
        
        self.device = torch.cuda.current_device()
        self.BERT = BERT.to(self.device)
        
#         for param in self.BERT.parameters():
#             param.requires_grad = False

        self.tokenizer = BERT_tokenizer
        bert_hidden_dim = self.BERT.pooler.dense.out_features
                                  
        self._passage_weights_predictor = ntorch.nn.Linear(bert_hidden_dim, 1)
        self._question_weights_predictor = ntorch.nn.Linear(bert_hidden_dim, 1)
        
        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FeedForward(2 * bert_hidden_dim,
                                                     activations=[ReLU(),
                                                                  Identity()],
                                                     hidden_dims=[bert_hidden_dim,
                                                                  len(self.answering_abilities)],
                                                     num_layers=2,
                                                     dropout=dropout)
        
        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = FeedForward(bert_hidden_dim,
                                                             activations=[Identity()],
                                                             hidden_dims=[1],
                                                             num_layers=1)
            self._passage_span_end_predictor = FeedForward(bert_hidden_dim,
                                                             activations=[Identity()],
                                                             hidden_dims=[1],
                                                             num_layers=1)
        
        if "question_span_extraction" in self.answering_abilities:                                                
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._question_span_start_predictor = FeedForward(bert_hidden_dim * 2,
                                                              activations=[ReLU(),
                                                                       Identity()],
                                                          hidden_dims=[bert_hidden_dim, 1],
                                                              num_layers=2)
       
            self._question_span_end_predictor = FeedForward(bert_hidden_dim * 2,
                                                              activations=[ReLU(),
                                                                           Identity()],
                                                              hidden_dims=[bert_hidden_dim, 1],
                                                              num_layers=2)
        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = FeedForward(bert_hidden_dim * 2,
                                                      activations=[ReLU(),
                                                                   Identity()],
                                                      hidden_dims=[bert_hidden_dim, 3],
                                                      num_layers=2)
        
        if "counting" in self.answering_abilities:                                                      
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = FeedForward(bert_hidden_dim,
                                                   activations=[ReLU(),
                                                                Identity()],
                                                   hidden_dims=[bert_hidden_dim, 10],
                                                   num_layers=2)
        
        self.dropout = ntorch.nn.Dropout(dropout)
        self._drop_metrics = DropEmAndF1()
        
        
        
    def summary_vector(self, encoding, mask, passage = True):
        if passage:
            alpha = self._passage_weights_predictor(encoding).stack(("seqlen","out"), "seqlen")
        else:
            alpha = self._question_weights_predictor(encoding).stack(("seqlen","out"), "seqlen")
        alpha = nhelpers.masked_softmax(alpha, mask, "seqlen")
        h = alpha.dot("seqlen", encoding)
        return h
        
    def forward(self, question_passage, mask_indices, number_indices, 
                answer_as_passage_spans = None,
                answer_as_question_spans = None,
                answer_as_add_sub_expressions = None,
                answer_as_counts = None,
                metadata = None):
        
        question_passage_tokens = question_passage["tokens"]
        pad_mask = question_passage["mask"] 
        seqlen_ids = question_passage["tokens-type-ids"]
        
        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]
        
        mask_1 = mask_indices.squeeze()
        mask_2 = (mask_1 < max_seqlen).long()
        mask = mask_1 * mask_2
        cls_seq_mask = torch.ones(pad_mask.shape).to(self.device).long().scatter(1, mask, torch.zeros(mask.shape).to(self.device).long())
        
        passage_mask = seqlen_ids * pad_mask * cls_seq_mask
        question_mask = (1 - seqlen_ids) * pad_mask * cls_seq_mask
        
        bert_out, _ = self.BERT(question_passage_tokens, seqlen_ids, pad_mask, output_all_encoded_layers=False)
        
        question_start = 0
        question_end = min(max(mask[:,1]), max_seqlen)
        question_out = ntorch.tensor(bert_out[:,question_start:question_end], names=("batch","seqlen","out"))
#         print(question_out.device)
#         print(self.summary_vector.device)
        question_mask = ntorch.tensor(question_mask[:,question_start:question_end], names=("batch","seqlen"))
        question_vector = self.summary_vector(question_out, question_mask, False)
        
        
        
        passage_out = ntorch.tensor(bert_out, names=("batch","seqlen","out"))
        del bert_out
        passage_mask = ntorch.tensor(passage_mask, names=("batch","seqlen"))
        passage_vector = self.summary_vector(passage_out, passage_mask)
        
        if len(self.answering_abilities) > 1:
            # get answer ability probs
            answer_ability_logits = \
                self._answer_ability_predictor(ntorch.cat([passage_vector, question_vector], dim="out"))
            answer_ability_log_probs = answer_ability_logits.log_softmax("out")
            best_answer_ability = answer_ability_log_probs.argmax("out")
        
        if "counting" in self.answering_abilities:
        # count
            count_number_logits = self._count_number_predictor(passage_vector)
            count_number_log_probs = count_number_logits.log_softmax("out")
            best_count_number = count_number_log_probs.argmax("out")
            best_count_log_prob = \
                    count_number_log_probs.gather("out",\
                                                  best_count_number.split("batch",("batch","out"), out=1),\
                                                  "out")
            best_count_log_prob = best_count_log_prob.stack(("batch","out"), "batch")
            if len(self.answering_abilities) > 1:
                best_count_log_prob = best_count_log_prob + answer_ability_log_probs[{"out":self._counting_index}]
        
        if "passage_span_extraction" in self.answering_abilities:
            # passage span
            passage_span_start_logits = self._passage_span_start_predictor(passage_out).\
                                            stack(("seqlen","out"),"seqlen")
            passage_span_start_log_probs = nhelpers.masked_log_softmax(passage_span_start_logits, passage_mask, "seqlen")                               
    #         passage_span_start_log_probs = passage_span_start_logits.log_softmax("seqlen")
            passage_span_start_log_probs = nhelpers.replace_masked_values(passage_span_start_log_probs, passage_mask, -1e7)

            passage_span_end_logits = self._passage_span_end_predictor(passage_out).stack(("seqlen","out"),"seqlen")
            passage_span_end_log_probs = nhelpers.masked_log_softmax(passage_span_end_logits, passage_mask, "seqlen")                                
    #         passage_span_end_log_probs = passage_span_end_logits.log_softmax("seqlen")
            passage_span_end_log_probs = nhelpers.replace_masked_values(passage_span_end_log_probs, passage_mask, -1e7)

            best_passage_start, best_passage_end  = \
                nhelpers.get_best_span(passage_span_start_logits, passage_span_end_logits)

            best_passage_start_log_probs = \
                passage_span_start_log_probs.gather("seqlen", \
                                                    best_passage_start.split("batch",("batch","seqlen"), seqlen=1),\
                                                    "seqlen")
            best_passage_start_log_probs = best_passage_start_log_probs.stack(("batch","seqlen"), "batch")

            best_passage_end_log_probs = \
                passage_span_end_log_probs.gather("seqlen", \
                                                  best_passage_end.split("batch",("batch","seqlen"), seqlen=1),\
                                                  "seqlen")
            best_passage_end_log_probs = best_passage_end_log_probs.stack(("batch","seqlen"), "batch")

            best_passage_span_log_prob = best_passage_start_log_probs + best_passage_end_log_probs
            
            if len(self.answering_abilities) > 1:
                best_passage_span_log_prob = best_passage_span_log_prob + answer_ability_log_probs[{"out":self._passage_span_extraction_index}]
        if "question_span_extraction" in self.answering_abilities:
            # question span
            encoded_question_for_span_prediction = \
                ntorch.cat([question_out, nhelpers.repeat(passage_vector.split("out", ("seqlen","out"), seqlen=1),\
                                                         {"seqlen":question_out.shape["seqlen"]})], "out")

            question_span_start_logits = \
                    self._question_span_start_predictor(encoded_question_for_span_prediction)
            question_span_start_logits = question_span_start_logits.stack(("seqlen","out"), "seqlen")
            question_span_start_log_probs = nhelpers.masked_log_softmax(question_span_start_logits, question_mask, "seqlen")

            question_span_end_logits = \
                    self._question_span_end_predictor(encoded_question_for_span_prediction)
            question_span_end_logits = question_span_end_logits.stack(("seqlen","out"), "seqlen")
            question_span_end_log_probs = nhelpers.masked_log_softmax(question_span_end_logits, question_mask, "seqlen")

            question_span_start_logits = \
                    nhelpers.replace_masked_values(question_span_start_logits, question_mask, -1e7)
            question_span_end_logits = \
                    nhelpers.replace_masked_values(question_span_end_logits, question_mask, -1e7)

            best_question_start, best_question_end  = \
                nhelpers.get_best_span(question_span_start_logits, question_span_end_logits)

            best_question_start_log_probs = \
                question_span_start_log_probs.gather("seqlen", \
                                                best_question_start.split("batch",("batch","seqlen"), seqlen=1),\
                                                    "seqlen")
            best_question_start_log_probs = best_question_start_log_probs.stack(("batch","seqlen"), "batch")

            best_question_end_log_probs = \
                question_span_end_log_probs.gather("seqlen", \
                                                  best_question_end.split("batch",("batch","seqlen"), seqlen=1),\
                                                  "seqlen")
            best_question_end_log_probs = best_question_end_log_probs.stack(("batch","seqlen"), "batch")

            best_question_span_log_prob = best_question_start_log_probs + best_question_end_log_probs
            
            if len(self.answering_abilities) > 1:
                best_question_span_log_prob = best_question_span_log_prob + answer_ability_log_probs[{"out":self._question_span_extraction_index}]
        
        if "addition_subtraction" in self.answering_abilities:
            number_indices = number_indices.squeeze()
            number_indices[number_indices >= max_seqlen] = -1
            number_indices = ntorch.tensor(number_indices, names=("batch","seqlen"))
            number_mask = (number_indices != -1).long()
            clamped_number_indices = nhelpers.replace_masked_values(number_indices, number_mask, 0)

            encoded_numbers = passage_out.gather("seqlen", \
                                                 nhelpers.repeat(clamped_number_indices.split("seqlen", ("seqlen","out"),\
                                                                                                out=1),\
                                                                 {"out":passage_out.shape["out"]}),\
                                                 "seqlen")

            encoded_numbers = \
                ntorch.cat([encoded_numbers, \
                            nhelpers.repeat(passage_vector.split("out", ("seqlen","out"), seqlen=1),\
                                             {"seqlen":encoded_numbers.shape["seqlen"]})],\
                           "out")

            number_sign_logits = self._number_sign_predictor(encoded_numbers)
            number_sign_log_probs = number_sign_logits.log_softmax("out")

            best_signs_for_numbers = number_sign_log_probs.argmax("out")
            best_signs_for_numbers = nhelpers.replace_masked_values(best_signs_for_numbers, number_mask, 0)

            best_signs_log_probs = \
                number_sign_log_probs.gather("out", \
                                             best_signs_for_numbers.split("seqlen",("seqlen","out"), out=1),\
                                             "out")
            best_signs_log_probs = best_signs_log_probs.stack(("seqlen","out"), "seqlen")
            best_signs_log_probs = nhelpers.replace_masked_values(best_signs_log_probs, number_mask, 0)

            best_combination_log_prob = best_signs_log_probs.sum("seqlen")
            
            if len(self.answering_abilities) > 1:
                best_combination_log_prob = best_combination_log_prob + answer_ability_log_probs[{"out":self._addition_subtraction_index}]
        
        output_dict = {}
        
        if answer_as_passage_spans is not None or answer_as_question_spans is not None \
                or answer_as_add_sub_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []
            
            for answering_ability in self.answering_abilities:
            
                if answering_ability == "passage_span_extraction":
                    answer_as_passage_spans[:,:][answer_as_passage_spans[:,:,1] >= max_seqlen] = -1
                    gold_passage_span_starts = ntorch.tensor(answer_as_passage_spans[:, :, 0],\
                                                             names=("batch","seqlen"))
                    gold_passage_span_ends = ntorch.tensor(answer_as_passage_spans[:, :, 1], \
                                                           names=("batch","seqlen"))

                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()

                    clamped_gold_passage_span_starts = \
                            nhelpers.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = \
                            nhelpers.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)

                    log_likelihood_for_passage_span_starts = \
                        passage_span_start_log_probs.gather("seqlen", clamped_gold_passage_span_starts, "seqlen")

                    log_likelihood_for_passage_span_ends = \
                        passage_span_end_log_probs.gather("seqlen", clamped_gold_passage_span_ends, "seqlen")

                    log_likelihood_for_passage_spans = \
                        log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends

                    log_likelihood_for_passage_spans = \
                            nhelpers.replace_masked_values(log_likelihood_for_passage_spans, \
                                                       gold_passage_span_mask, -1e7)

                    log_marginal_likelihood_for_passage_span = log_likelihood_for_passage_spans.logsumexp("seqlen")

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":

                    gold_question_span_starts = ntorch.tensor(answer_as_question_spans[:, :, 0],\
                                                             names=("batch","seqlen"))
                    gold_question_span_ends = ntorch.tensor(answer_as_question_spans[:, :, 1], \
                                                           names=("batch","seqlen"))

                    gold_question_span_mask = (gold_question_span_starts != -1).long()

                    clamped_gold_question_span_starts = \
                            nhelpers.replace_masked_values(gold_question_span_starts, gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = \
                            nhelpers.replace_masked_values(gold_question_span_ends, gold_question_span_mask, 0)

                    log_likelihood_for_question_span_starts = \
                        question_span_start_log_probs.gather("seqlen", clamped_gold_question_span_starts, "seqlen")

                    log_likelihood_for_question_span_ends = \
                        question_span_end_log_probs.gather("seqlen", clamped_gold_question_span_ends, "seqlen")

                    log_likelihood_for_question_spans = \
                        log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends

                    log_likelihood_for_question_spans = \
                        nhelpers.replace_masked_values(log_likelihood_for_question_spans, \
                                                       gold_question_span_mask, -1e7)

                    log_marginal_likelihood_for_question_span = log_likelihood_for_question_spans.logsumexp("seqlen")

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)

                elif answering_ability == "addition_subtraction":
                    answer_as_add_sub_expressions = ntorch.tensor(answer_as_add_sub_expressions, names=("batch","out","seqlen"))
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum("seqlen") > 0).float()
                    gold_add_sub_signs = answer_as_add_sub_expressions

                    log_likelihood_for_number_signs = \
                        number_sign_log_probs.gather("out", gold_add_sub_signs, "out")
                    log_likelihood_for_number_signs = \
                        nhelpers.replace_masked_values(log_likelihood_for_number_signs, number_mask.split("seqlen",("seqlen","out"), out=1), 0)

                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum("seqlen")
                    log_likelihood_for_add_subs = \
                        nhelpers.replace_masked_values(log_likelihood_for_add_subs, gold_add_sub_mask, -1e7)

                    log_marginal_likelihood_for_add_sub = log_likelihood_for_add_subs.logsumexp("out")

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)


                elif answering_ability == "counting":
                    answer_as_counts = ntorch.tensor(answer_as_counts, names=("batch","out"))
                    gold_count_mask = (answer_as_counts != -1).long()
                    clamped_gold_counts = nhelpers.replace_masked_values(answer_as_counts, gold_count_mask, 0)

                    log_likelihood_for_counts = count_number_log_probs.gather("out", clamped_gold_counts,"out")

                    log_likelihood_for_counts = \
                        nhelpers.replace_masked_values(log_likelihood_for_counts, gold_count_mask, -1e7)

                    log_marginal_likelihood_for_count = log_likelihood_for_counts.logsumexp("out")

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)
            if len(self.answering_abilities) > 1:
                lml_list2 = [lml.split("batch", ("batch","out"), out=1) for lml in log_marginal_likelihood_list]
                all_log_marginal_likelihoods = ntorch.cat(lml_list2, "out")
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = all_log_marginal_likelihoods.logsumexp("out")
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
            output_dict["loss"] = - marginal_log_likelihood.mean().values
#             output_dict["ability"] = answer_ability_log_probs.values
        
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            
            for i in range(batch_size):
                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[{"batch":i}].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]
                
                answer_json: Dict[str, Any] = {}
                
                if predicted_ability_str == "passage_span_extraction":
                    answer_json["answer_type"] = "passage_span"
                    predicted_start = best_passage_start[{"batch":i}].detach().cpu().item()
                    predicted_end = best_passage_end[{"batch":i}].detach().cpu().item()
                    answer_json["spans"] = [(predicted_start, predicted_end)]
                    answer_tokens = question_passage_tokens[i][predicted_start:predicted_end + 1].detach().cpu().numpy()
                    token_lst = self.tokenizer.convert_ids_to_tokens(answer_tokens)
                    predicted_answer = nhelpers.tokenlist_to_passage(token_lst)
                    answer_json["value"] = predicted_answer
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    predicted_start = best_question_start[{"batch":i}].detach().cpu().item()
                    predicted_end = best_question_end[{"batch":i}].detach().cpu().item()
                    answer_json["spans"] = [(predicted_start, predicted_end)]
                    answer_tokens = question_passage_tokens[i][predicted_start:predicted_end + 1].detach().cpu().numpy()
                    token_lst = self.tokenizer.convert_ids_to_tokens(answer_tokens)
                    predicted_answer = nhelpers.tokenlist_to_passage(token_lst)
                    answer_json["value"] = predicted_answer
                elif predicted_ability_str == "addition_subtraction":
                    answer_json["answer_type"] = "arithmetic"
                    original_numbers = metadata[i]['original_numbers']
                    sign_remap = {0: 0, 1: 1, 2: -1}
                    predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[{"batch":i}].detach().cpu().numpy()]
                    result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                    predicted_answer = str(result)
                    answer_json['numbers'] = []
                    for value, sign in zip(original_numbers, predicted_signs):
                        answer_json['numbers'].append({'value': value, 'sign': sign})
                    if number_indices.values[i,-1] == -1:
                        answer_json["numbers"].pop()
                    answer_json["value"] = result
                elif predicted_ability_str == "counting":
                    answer_json["answer_type"] = "count"
                    predicted_count = best_count_number[{"batch":i}].detach().cpu().item()
                    predicted_answer = str(predicted_count)
                    answer_json["count"] = predicted_count
                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations)
        return output_dict    

    def get_metrics(self, reset = False):
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}