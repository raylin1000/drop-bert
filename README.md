# drop_bert

This repository contains the code for our CS 287 spring 2019 final project.

The code requires pytorch and allennlp.

The `notebooks` folder contains notebooks for generating specs for the DROP dataset and metrics for a trained model, as well
as a sample notebook containing a sample `allennlp train` command we used to train our models.

The `drop_bert` folder contains our models. Files with `old` at the end are legacy code for backwards compatability with some
older models we trained, so those are probably not particularly useful. `augmented_bert.py` contains the NABERT base model, `augmented_bert_plus.py` contains the NABERT+ model with only standard numbers, and `augmented_bert_templated.py` contains the full NABERT+ model (with standard numbers as well as templating). `data_processing.py` contains the code for processing the raw DROP JSON files into forms that can be fed into our models.

The `config` folder contains the JSON configurations needed to run the models using `allennlp train`.
