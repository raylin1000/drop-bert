import torch
from namedtensor import ntorch
from activations import ReLU, Identity
from feedforward import FeedForward
from nhelpers import get_best_span
import nhelpers


class AugmentedBERT(ntorch.nn.Module):
    """Numerically Augemented BERT for DROP"""

    def __init__(
        self, BERT, answering_abilities=None, dropout=0.0
    ):
        super(AugmentedBERT, self).__init__()

        if answering_abilities is None:
            self.answering_abilities = [
                "passage_span_extraction",
                "question_span_extraction",
                "addition_subtraction",
                "counting",
            ]
        else:
            self.answering_abilities = answering_abilities

        self.BERT = BERT

        bert_hidden_dim = (
            self.BERT.pooler.dense.out_features
        )

        self._passage_weights_predictor = ntorch.nn.Linear(
            bert_hidden_dim, 1
        )
        self._question_weights_predictor = ntorch.nn.Linear(
            bert_hidden_dim, 1
        )

        self._answer_ability_predictor = FeedForward(
            2 * bert_hidden_dim,
            activations=[ReLU(), Identity()],
            hidden_dims=[
                bert_hidden_dim,
                len(self.answering_abilities),
            ],
            num_layers=2,
            dropout=dropout,
        )

        self._passage_span_extraction_index = self.answering_abilities.index(
            "passage_span_extraction"
        )
        self._passage_span_start_predictor = FeedForward(
            bert_hidden_dim,
            activations=[Identity()],
            hidden_dims=[1],
            num_layers=1,
        )
        self._passage_span_end_predictor = FeedForward(
            bert_hidden_dim,
            activations=[Identity()],
            hidden_dims=[1],
            num_layers=1,
        )

        self._question_span_extraction_index = self.answering_abilities.index(
            "question_span_extraction"
        )
        self._question_span_start_predictor = FeedForward(
            bert_hidden_dim * 2,
            activations=[ReLU(), Identity()],
            hidden_dims=[bert_hidden_dim, 1],
            num_layers=2,
        )
        self._question_span_end_predictor = FeedForward(
            bert_hidden_dim * 2,
            activations=[ReLU(), Identity()],
            hidden_dims=[bert_hidden_dim, 1],
            num_layers=2,
        )

        self._addition_subtraction_index = self.answering_abilities.index(
            "addition_subtraction"
        )
        self._number_sign_predictor = FeedForward(
            bert_hidden_dim * 2,
            activations=[ReLU(), Identity()],
            hidden_dims=[bert_hidden_dim, 3],
            num_layers=2,
        )

        self._counting_index = self.answering_abilities.index(
            "counting"
        )
        self._count_number_predictor = FeedForward(
            bert_hidden_dim,
            activations=[ReLU(), Identity()],
            hidden_dims=[bert_hidden_dim, 10],
            num_layers=2,
        )

        self.dropout = ntorch.nn.Dropout(dropout)

    def summary_vector(self, encoding, passage=True):
        if passage:
            alpha = self._passage_weights_predictor(
                encoding
            ).stack(("seqlen", "out"), "seqlen")
        else:
            alpha = self._question_weights_predictor(
                encoding
            ).stack(("seqlen", "out"), "seqlen")
        alpha = alpha.softmax("seqlen")
        h = alpha.dot("seqlen", encoding)
        return h

    def forward(
        self,
        question,
        passage,
        number_indices,
        answer_as_passage_spans=None,
        answer_as_question_spans=None,
        answer_as_add_sub_expressions=None,
        answer_as_counts=None,
    ):
        qlen = question.shape[-1]
        plen = passage.shape[-1]

        # question_mask = util.get_text_field_mask(question).float()
        # passage_mask = util.get_text_field_mask(passage).float()

        bert_in = torch.cat([question, passage], dim=-1)
        seg_ids = (
            torch.cat([torch.zeros(qlen), torch.ones(plen)])
            .long()
            .cuda()
        )

        bert_out, _ = model(
            bert_in,
            seg_ids,
            output_all_encoded_layers=False,
        )

        question_out = ntorch.tensor(
            bert_out[:, :qlen],
            names=("batch", "seqlen", "out"),
        )
        question_vector = self.summary_vector(
            question_out, False
        )

        passage_out = ntorch.tensor(
            bert_out[:, qlen:],
            names=("batch", "seqlen", "out"),
        )
        passage_vector = self.summary_vector(passage_out)

        # get answer ability probs
        answer_ability_logits = self._answer_ability_predictor(
            ntorch.cat(
                [passage_vector, question_vector], dim="out"
            )
        )
        answer_ability_log_probs = answer_ability_logits.log_softmax(
            "out"
        )
        best_answer_ability = answer_ability_log_probs.argmax(
            "out"
        )

        # count
        count_number_logits = self._count_number_predictor(
            passage_vector
        )
        count_number_log_probs = count_number_logits.log_softmax(
            "out"
        )
        best_count_number = count_number_log_probs.argmax(
            "out"
        )
        best_count_log_prob = count_number_log_probs.gather(
            "out",
            best_count_number.split(
                "batch", ("batch", "out"), out=1
            ),
            "out",
        )
        best_count_log_prob = best_count_log_prob.stack(
            ("batch", "out"), "batch"
        )
        best_count_log_prob += answer_ability_log_probs[
            {"out": self._counting_index}
        ]

        # passage span
        passage_span_start_logits = self._passage_span_start_predictor(
            passage_out
        ).stack(
            ("seqlen", "out"), "seqlen"
        )
        passage_span_start_log_probs = passage_span_start_logits.log_softmax(
            "seqlen"
        )
        passage_span_start_log_probs = nhelpers.replace_masked_values(
            passage_span_start_log_probs, passage_mask, -1e7
        )

        passage_span_end_logits = self._passage_span_end_predictor(
            passage_out
        ).stack(
            ("seqlen", "out"), "seqlen"
        )
        passage_span_end_log_probs = passage_span_end_logits.log_softmax(
            "seqlen"
        )
        passage_span_end_log_probs = nhelpers.replace_masked_values(
            passage_span_end_log_probs, passage_mask, -1e7
        )

        best_passage_start, best_passage_end = get_best_span(
            passage_span_start_logits,
            passage_span_end_logits,
        )

        best_passage_start_log_probs = passage_span_start_log_probs.gather(
            "seqlen",
            best_passage_start.split(
                "batch", ("batch", "seqlen"), seqlen=1
            ),
            "seqlen",
        )
        best_passage_start_log_probs = best_passage_start_log_probs.stack(
            ("batch", "seqlen"), "batch"
        )

        best_passage_end_log_probs = passage_span_end_log_probs.gather(
            "seqlen",
            best_passage_end.split(
                "batch", ("batch", "seqlen"), seqlen=1
            ),
            "seqlen",
        )
        best_passage_end_log_probs = best_passage_end_log_probs.stack(
            ("batch", "seqlen"), "batch"
        )

        best_passage_span_log_prob = (
            best_passage_start_log_probs
            + best_passage_end_log_probs
        )
        best_passage_span_log_prob += answer_ability_log_probs[
            {"out": self._passage_span_extraction_index}
        ]

        # question span
        encoded_question_for_span_prediction = ntorch.cat(
            [
                question_out,
                nhelpers.repeat(
                    passage_vector.split(
                        "out", ("seqlen", "out"), seqlen=1
                    ),
                    {
                        "seqlen": question_out.shape[
                            "seqlen"
                        ]
                    },
                ),
            ],
            "out",
        )

        question_span_start_logits = self._question_span_start_predictor(
            encoded_question_for_span_prediction
        )
        question_span_start_logits = question_span_start_logits.stack(
            ("seqlen", "out"), "seqlen"
        )
        question_span_start_log_probs = nhelpers.masked_log_softmax(
            question_span_start_logits,
            question_mask,
            "seqlen",
        )

        question_span_end_logits = self._question_span_end_predictor(
            encoded_question_for_span_prediction
        )
        question_span_end_logits = question_span_end_logits.stack(
            ("seqlen", "out"), "seqlen"
        )
        question_span_end_log_probs = nhelpers.masked_log_softmax(
            question_span_end_logits,
            question_mask,
            "seqlen",
        )

        question_span_start_logits = nhelpers.replace_masked_values(
            question_span_start_logits, question_mask, -1e7
        )
        question_span_end_logits = nhelpers.replace_masked_values(
            question_span_end_logits, question_mask, -1e7
        )

        best_question_start, best_question_end = get_best_span(
            question_span_start_logits,
            question_span_end_logits,
        )

        best_question_start_log_probs = question_span_start_log_probs.gather(
            "seqlen",
            best_question_start.split(
                "batch", ("batch", "seqlen"), seqlen=1
            ),
            "seqlen",
        )
        best_question_start_log_probs = best_question_start_log_probs.stack(
            ("batch", "seqlen"), "batch"
        )

        best_question_end_log_probs = question_span_end_log_probs.gather(
            "seqlen",
            best_question_end.split(
                "batch", ("batch", "seqlen"), seqlen=1
            ),
            "seqlen",
        )
        best_question_end_log_probs = best_question_end_log_probs.stack(
            ("batch", "seqlen"), "batch"
        )

        best_question_span_log_prob = (
            best_question_start_log_probs
            + best_question_end_log_probs
        )
        best_question_span_log_prob += answer_ability_log_probs[
            {"out": self._question_span_extraction_index}
        ]

        number_indices = ntorch.tensor(
            number_indices, names=("batch", "seqlen")
        )
        number_mask = (number_indices != -1).long()
        clamped_number_indices = nhelpers.replace_masked_values(
            number_indices, number_mask, 0
        )

        encoded_numbers = passage_out.gather(
            "seqlen",
            nhelpers.repeat(
                clamped_number_indices.split(
                    "seqlen", ("seqlen", "out"), out=1
                ),
                {"out": passage_out.shape["out"]},
            ),
            "seqlen",
        )

        encoded_numbers = ntorch.cat(
            [
                encoded_numbers,
                nhelpers.repeat(
                    passage_vector.split(
                        "out", ("seqlen", "out"), seqlen=1
                    ),
                    {
                        "seqlen": encoded_numbers.shape[
                            "seqlen"
                        ]
                    },
                ),
            ],
            "out",
        )

        number_sign_logits = self._number_sign_predictor(
            encoded_numbers
        )
        number_sign_log_probs = number_sign_logits.log_softmax(
            "out"
        )

        best_signs_for_numbers = number_sign_log_probs.argmax(
            "out"
        )
        best_signs_for_numbers = nhelpers.replace_masked_values(
            best_signs_for_numbers, number_mask, 0
        )

        best_signs_log_probs = number_sign_log_probs.gather(
            "out",
            best_signs_for_numbers.split(
                "seqlen", ("seqlen", "out"), out=1
            ),
            "out",
        )
        best_signs_log_probs = best_signs_log_probs.stack(
            ("seqlen", "out"), "seqlen"
        )
        best_signs_log_probs = nhelpers.replace_masked_values(
            best_signs_log_probs, number_mask, 0
        )

        best_combination_log_prob = best_signs_log_probs.sum(
            "seqlen"
        )

        best_combination_log_prob += answer_ability_log_probs[
            {"out": self._addition_subtraction_index}
        ]

        if (
            answer_as_passage_spans is not None
            or answer_as_question_spans is not None
            or answer_as_add_sub_expressions is not None
            or answer_as_counts is not None
        ):

            log_marginal_likelihood_list = []

            for (
                answering_ability
            ) in self.answering_abilities:

                if (
                    answering_ability
                    == "passage_span_extraction"
                ):
                    gold_passage_span_starts = ntorch.tensor(
                        answer_as_passage_spans[:, :, 0],
                        names=("batch", "seqlen"),
                    )
                    gold_passage_span_ends = ntorch.tensor(
                        answer_as_passage_spans[:, :, 1],
                        names=("batch", "seqlen"),
                    )

                    gold_passage_span_mask = (
                        gold_passage_span_starts != -1
                    ).long()

                    clamped_gold_passage_span_starts = nhelpers.replace_masked_values(
                        gold_passage_span_starts,
                        gold_passage_span_mask,
                        0,
                    )
                    clamped_gold_passage_span_ends = nhelpers.replace_masked_values(
                        gold_passage_span_ends,
                        gold_passage_span_mask,
                        0,
                    )

                    log_likelihood_for_passage_span_starts = passage_span_start_log_probs.gather(
                        "seqlen",
                        clamped_gold_passage_span_starts,
                        "seqlen",
                    )

                    log_likelihood_for_passage_span_ends = passage_span_end_log_probs.gather(
                        "seqlen",
                        clamped_gold_passage_span_ends,
                        "seqlen",
                    )

                    log_likelihood_for_passage_spans = (
                        log_likelihood_for_passage_span_starts
                        + log_likelihood_for_passage_span_ends
                    )

                    log_likelihood_for_passage_spans = nhelpers.replace_masked_values(
                        log_likelihood_for_passage_spans,
                        gold_passage_span_mask,
                        -1e7,
                    )

                    log_marginal_likelihood_for_passage_span = log_likelihood_for_passage_spans.logsumexp(
                        "seqlen"
                    )

                    log_marginal_likelihood_list.append(
                        log_marginal_likelihood_for_passage_span
                    )

                elif (
                    answering_ability
                    == "question_span_extraction"
                ):

                    gold_question_span_starts = ntorch.tensor(
                        answer_as_question_spans[:, :, 0],
                        names=("batch", "seqlen"),
                    )
                    gold_question_span_ends = ntorch.tensor(
                        answer_as_question_spans[:, :, 1],
                        names=("batch", "seqlen"),
                    )

                    gold_question_span_mask = (
                        gold_question_span_starts != -1
                    ).long()

                    clamped_gold_question_span_starts = nhelpers.replace_masked_values(
                        gold_question_span_starts,
                        gold_question_span_mask,
                        0,
                    )
                    clamped_gold_question_span_ends = nhelpers.replace_masked_values(
                        gold_question_span_ends,
                        gold_question_span_mask,
                        0,
                    )

                    log_likelihood_for_question_span_starts = question_span_start_log_probs.gather(
                        "seqlen",
                        clamped_gold_question_span_starts,
                        "seqlen",
                    )

                    log_likelihood_for_question_span_ens = question_span_end_log_probs.gather(
                        "seqlen",
                        clamped_gold_question_span_ends,
                        "seqlen",
                    )

                    log_likelihood_for_question_spans = (
                        log_likelihood_for_question_span_starts
                        + log_likelihood_for_question_span_ends
                    )

                    log_likelihood_for_question_spans = nhelpers.replace_masked_values(
                        log_likelihood_for_question_spans,
                        gold_question_span_mask,
                        -1e7,
                    )

                    log_marginal_likelihood_for_question_span = log_likelihood_for_question_spans.logsumexp(
                        "seqlen"
                    )

                    log_marginal_likelihood_list.append(
                        log_marginal_likelihood_for_question_span
                    )

                elif (
                    answering_ability
                    == "addition_subtraction"
                ):
                    answer_as_add_sub_expressions = ntorch.tensor(
                        answer_as_add_sub_expressions,
                        names=("batch", "out", "seqlen"),
                    )
                    gold_add_sub_mask = (
                        answer_as_add_sub_expressions.sum(
                            "seqlen"
                        )
                        > 0
                    ).float()
                    gold_add_sub_signs = (
                        answer_as_add_sub_expressions
                    )

                    log_likelihood_for_number_signs = torch.gather(
                        number_sign_log_probs,
                        2,
                        gold_add_sub_signs,
                    )
                    log_likelihood_for_number_signs = number_sign_log_probs.gather(
                        "out", gold_add_sub_signs, "out"
                    )
                    log_likelihood_for_number_signs = nhelpers.replace_masked_values(
                        log_likelihood_for_number_signs,
                        number_mask.split(
                            "seqlen",
                            ("seqlen", "out"),
                            out=1,
                        ),
                        0,
                    )

                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(
                        "seqlen"
                    )
                    log_likelihood_for_add_subs = nhelpers.replace_masked_values(
                        log_likelihood_for_add_subs,
                        gold_add_sub_mask,
                        -1e7,
                    )

                    nlog_marginal_likelihood_for_add_sub = nlog_likelihood_for_add_subs.logsumexp(
                        "out"
                    )

                    log_marginal_likelihood_list.append(
                        log_marginal_likelihood_for_add_sub
                    )

                elif answering_ability == "counting":
                    answer_as_counts = ntorch.tensor(
                        answer_as_counts,
                        names=("batch", "out"),
                    )
                    gold_count_mask = (
                        answer_as_counts != -1
                    ).long()
                    clamped_gold_counts = nhelpers.replace_masked_values(
                        answer_as_counts, gold_count_mask, 0
                    )

                    log_likelihood_for_counts = count_number_log_probs.gather(
                        "out", clamped_gold_counts, "out"
                    )

                    log_likelihood_for_counts = nhelpers.replace_masked_values(
                        log_likelihood_for_counts,
                        gold_count_mask,
                        -1e7,
                    )

                    log_marginal_likelihood_for_count = nlog_likelihood_for_counts.logsumexp(
                        "out"
                    )

                    log_marginal_likelihood_list.append(
                        log_marginal_likelihood_for_count
                    )

            lml_list2 = [
                lml.split("batch", ("batch", "out"), out=1)
                for lml in log_marginal_likelihood_list
            ]
            all_log_marginal_likelihoods = torch.stack(
                lml_list2, "out"
            )
            all_log_marginal_likelihoods = (
                all_log_marginal_likelihoods
                + answer_ability_log_probs
            )
            marginal_log_likelihood = all_log_marginal_likelihoods.logsumexp(
                "out"
            )

        return all_log_marginal_likelihoods.mean()
