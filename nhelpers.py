from namedtensor import NamedTensor


def repeat(tensor, dims_sizes):
    names = tensor._schema._names
    shape = tensor.shape

    new_sizes = []
    for name in names:
        if name in dims_sizes:
            new_sizes.append(dims_sizes[name])
        else:
            new_sizes.append(1)
    return NamedTensor(
        tensor._tensor.repeat(new_sizes), names
    )


def replace_masked_values(tensor, mask, replace_with):
    if tensor.dim() != mask.dim():
        raise ConfigurationError(
            "tensor.dim() (%d) != mask.dim() (%d)"
            % (tensor.dim(), mask.dim())
        )
    return tensor.masked_fill_(
        (1 - mask).byte(), replace_with
    )


def masked_softmax(
    vector,
    mask,
    dim,
    memory_efficient=True,
    mask_fill_value=-1e32,
):
    mask = mask.float()
    if not memory_efficient:
        result = (vector * mask).softmax(dim)
        result = result * mask
        result = result / (result.sum(dim) + 1e-13)
    else:
        masked_vector = vector.masked_fill_(
            (1 - mask).byte(), mask_fill_value
        )
        result = masked_vector.softmax(dim)
    return result


def masked_log_softmax(vector, mask, dim):
    mask = mask.float()
    vector = vector + (mask + 1e-45).log()
    return vector.log_softmax(dim)


def get_best_span(span_start_logits, span_end_logits):
    batch_size = span_start_logits.shape["batch"]
    passage_length = span_start_logits.shape["seqlen"]

    span_start_logits = span_start_logits.rename(
        "seqlen", "sseqlen"
    )
    span_end_logits = span_end_logits.rename(
        "seqlen", "eseqlen"
    )

    span_log_probs = span_start_logits + span_end_logits

    span_log_mask = (
        ntorch.triu(
            ntorch.ones(
                passage_length,
                passage_length,
                names=("sseqlen", "eseqlen"),
            ),
            dims=("sseqlen", "eseqlen"),
        )
        .log()
        .cuda()
    )
    valid_span_log_probs = span_log_probs + span_log_mask

    best_spans = valid_span_log_probs.stack(
        ("sseqlen", "eseqlen"), "seqlen"
    ).argmax("seqlen")

    span_start_indices = best_spans / passage_length
    span_end_indices = ntorch.fmod(
        best_spans, passage_length
    )

    return span_start_indices, span_end_indices
