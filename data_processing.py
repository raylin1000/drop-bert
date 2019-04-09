import json
from overrides import overrides
from tqdm import tqdm
from typing import Dict, List, Union, Tuple, Any

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from allennlp.data.fields import Field, TextField, IndexField, LabelField, ListField, \
                                 MetadataField, SequenceLabelField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from pytorch_pretrained_bert import BertTokenizer

class BertDropTokenizer(Tokenizer):
    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
    
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(token) for token in self.tokenizer.tokenize(text)]   

class BertDropReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,):
        super(BertDropReader, self).__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        
    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            
        instances = []
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"].strip()
            passage_tokens = self.tokenizer.tokenize(passage_text)
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer = question_answer['answer'] 
                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 question_id,
                                                 passage_id,
                                                 answer,
                                                 passage_tokens)
                if instance is not None:
                    instances.append(instance)
        return instances
                
    @overrides
    def text_to_instance(self, 
                         question_text: str, 
                         passage_text: str,
                         question_id: str = None, 
                         passage_id: str = None,
                         answers: Dict = None,
                         passage_tokens: List[Token] = None) -> Union[Instance, None]:
        # Tokenize question and passage
        question_tokens = self.tokenizer.tokenize(question_text)
        if not passage_tokens:
            passage_tokens = self.tokenizer.tokenize(passage_text)
            
        # Combine question and passage with separator
        question_concat_passage_tokens = [Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens + [Token('[SEP]')]
            
        # Get all the numbers in the passage
        numbers_in_passage = []
        number_indices = []
        for token_index, token in enumerate(passage_tokens):
            number = DropReader.convert_word_to_number(token.text)
            if number is not None:
                numbers_in_passage.append(number)
                number_indices.append(token_index)
        numbers_in_passage.append(0)
        number_indices.append(-1)
        number_tokens = [Token(str(number)) for number in numbers_in_passage]
        
        fields: Dict[str, Field] = {}
            
        # Add feature fields
        passage_field = TextField(passage_tokens, self.token_indexers)
        question_field = TextField(question_tokens, self.token_indexers)
        question_and_passage_field = TextField(question_concat_passage_tokens, self.token_indexers)      
        fields["passage"] = passage_field
        fields["question"] = question_field
        fields['question_and_passage'] = question_and_passage_field
        number_index_fields: List[Field] = [IndexField(index, passage_field) for index in number_indices]
        fields["number_indices"] = ListField(number_index_fields)
        numbers_in_passage_field = TextField(number_tokens, self.token_indexers)
        
        # Compile question, passage, answer metadata
        metadata = {"original_passage": passage_text,
                    "original_question": question_text,
                    "original_numbers": numbers_in_passage,
                    "original_answers": answers,
                    "passage_id": passage_id,
                    "question_id": question_id}
        fields["metadata"] = MetadataField(metadata)
        
        if answers:
            # Get answer type, answer text, tokenize
            answer_type, answer_texts = DropReader.extract_answer_info_from_annotation(answers)
            tokenized_answer_texts = []
            for answer_text in answer_texts:
                answer_tokens = self.tokenizer.tokenize(answer_text)
                tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens)) 
        
            # Find answer text in question
            valid_question_spans = DropReader.find_valid_spans(question_tokens, tokenized_answer_texts)
            for span_index, span in enumerate(valid_question_spans):
                valid_question_spans[span_index] = (span[0] + 1, span[1] + 1)

            # Find answer text in passage
            question_len = len(question_tokens)
            valid_passage_spans = DropReader.find_valid_spans(passage_tokens, tokenized_answer_texts)
            for span_index, span in enumerate(valid_passage_spans):
                valid_passage_spans[span_index] = (span[0] + question_len + 2, span[1] + question_len + 2)
        
            # Get target numbers
            target_numbers = []
            for answer_text in answer_texts:
                number = DropReader.convert_word_to_number(answer_text)
                if number is not None:
                    target_numbers.append(number)

            # Get possible ways to arrive at target numbers with add/sub        
            valid_signs_for_add_sub_expr: List[List[int]] = []
            if answer_type in ["number", "date"]:
                valid_signs_for_add_sub_expr = DropReader.find_valid_add_sub_expressions(numbers_in_passage, target_numbers)
            
            # Get possible ways to arrive at target numbers with counting
            valid_counts: List[int] = []
            if answer_type in ["number"]:
                # Only support count number 0 - 9
                numbers_for_count = list(range(10))
                valid_counts = DropReader.find_valid_counts(numbers_for_count, target_numbers)
        
            # Add answer fields
            passage_span_fields: List[Field] = [SpanField(span[0], span[1], question_and_passage_field) for span in valid_passage_spans]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, question_and_passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = [SpanField(span[0], span[1], question_and_passage_field) for span in valid_question_spans]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, question_and_passage_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            add_sub_signs_field: List[Field] = []
            for signs_for_one_add_sub_expr in valid_signs_for_add_sub_expr:
                add_sub_signs_field.append(SequenceLabelField(signs_for_one_add_sub_expr, numbers_in_passage_field))
            if not add_sub_signs_field:
                add_sub_signs_field.append(SequenceLabelField([0] * len(number_tokens), numbers_in_passage_field))
            fields["answer_as_add_sub_expr"] = ListField(add_sub_signs_field)

            count_fields: List[Field] = [LabelField(count_label, skip_indexing=True) for count_label in valid_counts]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)
        
        return Instance(fields)
