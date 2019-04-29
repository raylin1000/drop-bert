import json
from overrides import overrides
from tqdm import tqdm
from typing import Dict, List, Union, Tuple, Any
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.dataset_readers.reading_comprehension.util import split_tokens_by_hyphen
from allennlp.data.fields import Field, TextField, IndexField, LabelField, ListField, \
                                 MetadataField, SequenceLabelField, SpanField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer


from pytorch_pretrained_bert import BertTokenizer

from drop_nmn.nhelpers import tokenlist_to_passage, get_number_from_word

@Tokenizer.register("bert-drop")
class BertDropTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(token) for token in self.tokenizer.tokenize(text)]

@TokenIndexer.register("bert-drop")
class BertDropTokenIndexer(WordpieceIndexer):
    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512) -> None:
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         max_pieces=max_pieces,
                         namespace="bert",
                         separator_token="[SEP]")
    
   
@DatasetReader.register("bert-drop")
class BertDropReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_pieces: int = 512,
                 max_count: int = 10,
                 max_spans: int = 10,
                 max_numbers_expression: int = 2,
                 answer_type: List[str] = None,
                 use_validated: bool = True,
                 wordpiece_numbers: bool = True,
                 number_tokenizer: Tokenizer = None,
                 custom_word_to_num: bool = True):
        super(BertDropReader, self).__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_pieces = max_pieces
        self.max_count = max_count
        self.max_spans = max_spans
        self.max_numbers_expression = max_numbers_expression
        self.answer_type = answer_type
        self.use_validated = use_validated
        self.wordpiece_numbers = wordpiece_numbers
        self.number_tokenizer = number_tokenizer or WordTokenizer()
        if custom_word_to_num:
            self.word_to_num = get_number_from_word
        else:
            self.word_to_num = DropReader.convert_word_to_number
    
    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            
        instances = []
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"].strip()
            
            if self.wordpiece_numbers:        
                word_tokens = split_tokens_by_hyphen(self.number_tokenizer.tokenize(passage_text))
            else:
                word_tokens = self.tokenizer.tokenize(passage_text)
            numbers_in_passage = []
            number_indices = []
            number_words = []
            number_len = []
            passage_tokens = []
            curr_index = 0
            for token in word_tokens:
                number = self.word_to_num(token.text)
                wordpieces = self.tokenizer.tokenize(token.text)
                num_wordpieces = len(wordpieces)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(curr_index)
                    number_words.append(token.text)
                    number_len.append(num_wordpieces)
                passage_tokens += wordpieces
                curr_index += num_wordpieces
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    if self.answer_type is not None and self._get_answer_type(question_answer['answer']) not in self.answer_type:
                        continue
                    answer_annotations.append(question_answer["answer"])
                if self.use_validated and "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]
                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 passage_tokens,
                                                 numbers_in_passage,
                                                 number_words,
                                                 number_indices,
                                                 number_len,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations)
                if instance is not None:
                    instances.append(instance)
        return instances
                
    @overrides
    def text_to_instance(self, 
                         question_text: str, 
                         passage_text: str,
                         passage_tokens: List[Token],
                         numbers_in_passage: List[Any],
                         number_words : List[str],
                         number_indices: List[int],
                         number_len: List[int],
                         question_id: str = None, 
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         ) -> Union[Instance, None]:
        # Tokenize question and passage
        question_tokens = self.tokenizer.tokenize(question_text)
        qlen = len(question_tokens)
        plen = len(passage_tokens)

        question_passage_tokens = [Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens
        if len(question_passage_tokens) > self.max_pieces - 1:
            question_passage_tokens = question_passage_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - qlen - 3]
            plen = len(passage_tokens)
            number_indices, number_len, numbers_in_passage = \
                self._clipped_passage_num(number_indices, number_len, numbers_in_passage, plen)
        
        question_passage_tokens += [Token('[SEP]')]
        number_indices = [index + qlen + 2 for index in number_indices] + [-1]
        # Not done in-place so they won't change the numbers saved for the passage
        number_len = number_len + [1]
        numbers_in_passage = numbers_in_passage + [0]
        number_tokens = [Token(str(number)) for number in numbers_in_passage]
        
        mask_indices = [0, qlen + 1, len(question_passage_tokens) - 1]
        
        fields: Dict[str, Field] = {}
            
        # Add feature fields
        question_passage_field = TextField(question_passage_tokens, self.token_indexers)
        fields["question_passage"] = question_passage_field
       
        number_token_indices = \
            [ArrayField(np.arange(start_ind, start_ind + number_len[i]), padding_value=-1) 
             for i, start_ind in enumerate(number_indices)]
        fields["number_indices"] = ListField(number_token_indices)
        numbers_in_passage_field = TextField(number_tokens, self.token_indexers)
        mask_index_fields: List[Field] = [IndexField(index, question_passage_field) for index in mask_indices]
        fields["mask_indices"] = ListField(mask_index_fields)
        
        # Compile question, passage, answer metadata
        metadata = {"original_passage": passage_text,
                    "original_question": question_text,
                    "original_numbers": numbers_in_passage,
                    "original_number_words": number_words,
                    "passage_tokens": passage_tokens,
                    "question_tokens": question_tokens,
                    "question_passage_tokens": question_passage_tokens,
                    "passage_id": passage_id,
                    "question_id": question_id}
        
        
        if answer_annotations:
            for annotation in answer_annotations:
                tokenized_spans = [[token.text for token in self.tokenizer.tokenize(answer)] for answer in annotation['spans']]
                annotation['spans'] = [tokenlist_to_passage(token_list) for token_list in tokenized_spans]
            
            # Get answer type, answer text, tokenize
            answer_type, answer_texts = DropReader.extract_answer_info_from_annotation(answer_annotations[0])
            tokenized_answer_texts = []
            num_spans = min(len(answer_texts), self.max_spans)
            for answer_text in answer_texts:
                answer_tokens = self.tokenizer.tokenize(answer_text)
                tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens))
            
        
            metadata["answer_annotations"] = answer_annotations
            metadata["answer_texts"] = answer_texts
            metadata["answer_tokens"] = tokenized_answer_texts
            
            # Find answer text in question and passage
            valid_question_spans = DropReader.find_valid_spans(question_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_question_spans):
                valid_question_spans[span_ind] = (span[0] + 1, span[1] + 1)
            valid_passage_spans = DropReader.find_valid_spans(passage_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_passage_spans):
                valid_passage_spans[span_ind] = (span[0] + qlen + 2, span[1] + qlen + 2)
        
            # Get target numbers
            target_numbers = []
            for answer_text in answer_texts:
                number = self.word_to_num(answer_text)
                if number is not None:
                    target_numbers.append(number)

            # Get possible ways to arrive at target numbers with add/sub        
            valid_signs_for_add_sub_expressions: List[List[int]] = []
            if answer_type in ["number", "date"]:
                valid_signs_for_add_sub_expressions = \
                    DropReader.find_valid_add_sub_expressions(numbers_in_passage, target_numbers, self.max_numbers_expression)
            
            # Get possible ways to arrive at target numbers with counting
            valid_counts: List[int] = []
            if answer_type in ["number"]:
                numbers_for_count = list(range(self.max_count + 1))
                valid_counts = DropReader.find_valid_counts(numbers_for_count, target_numbers)
            
            # Update metadata with answer info
            answer_info = {"answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "num_spans": num_spans,
                           "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions,
                           "counts": valid_counts}
            metadata["answer_info"] = answer_info
        
            # Add answer fields
            passage_span_fields: List[Field] = [SpanField(span[0], span[1], question_passage_field) for span in valid_passage_spans]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, question_passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = [SpanField(span[0], span[1], question_passage_field) for span in valid_question_spans]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, question_passage_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            add_sub_signs_field: List[Field] = []
            for signs_for_one_add_sub_expressions in valid_signs_for_add_sub_expressions:
                add_sub_signs_field.append(SequenceLabelField(signs_for_one_add_sub_expressions, numbers_in_passage_field))
            if not add_sub_signs_field:
                add_sub_signs_field.append(SequenceLabelField([0] * len(number_tokens), numbers_in_passage_field))
            fields["answer_as_add_sub_expressions"] = ListField(add_sub_signs_field)

            count_fields: List[Field] = [LabelField(count_label, skip_indexing=True) for count_label in valid_counts]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)
            
            fields["num_spans"] = LabelField(num_spans, skip_indexing=True)
        
        fields["metadata"] = MetadataField(metadata)
        
        return Instance(fields)
    
    def _get_answer_type(self, answers):
        if answers['number']:
            return 'number'
        elif answers['spans']:
            if len(answers['spans']) == 1:
                return 'single_span'
            return 'multiple_span'
        elif any(answers['date'].values()):
            return 'date'
    
    def _clipped_passage_num(self, number_indices, number_len, numbers_in_passage, plen):
        if number_indices[-1] < plen:
            return number_indices, number_len, numbers_in_passage
        lo = 0
        hi = len(number_indices) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if number_indices[mid] < plen:
                lo = mid + 1
            else:
                hi = mid
        if number_indices[lo - 1] + number_len[lo - 1] > plen:
            number_len[lo - 1] = plen - number_indices[lo - 1]
        return number_indices[:lo], number_len[:lo], numbers_in_passage[:lo]
