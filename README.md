# drop_bert

This repository contains the code for our CS 287 spring 2019 final project.

The code requires `allennlp` version `0.8.3`, which can be installed by running `pip install allennlp==0.8.3`. When we set up our environment, we followed the instructions at the allennlp repo [here](https://github.com/allenai/allennlp).

The `notebooks` folder contains notebooks for generating specs for the DROP dataset and metrics for a trained model, as well
as a sample notebook containing a sample `allennlp train` command we used to train our models.

The `drop_bert` folder contains our models.  
- `augmented_bert.py` contains the NABERT base model
- `augmented_bert_plus.py` contains the NABERT+ model with only standard numbers
- `augmented_bert_templated.py` contains the full NABERT+ model (with standard numbers as well as templating) 
- `data_processing.py` contains the code for processing the raw DROP JSON files into forms that can be fed into our models.

The `config` folder contains the JSON configurations needed to run the models using `allennlp train`. We assume that the train and dev data are in `data/drop_dataset_train.json` and `data/drop_dataset_dev.json`, respectively, except for the pickle example, where we expect `pkl` files instead of `json`.
- `naqanet.json` contains the config we used to train a version of the NAQANet model
- `nabert.json` contains a config for the basic NABERT model
- `nabert-pickle.json` contains a config for the basic NABERT model, except where a dataset reader has already been used to read in the data to allennlp `Instances` and the result from there saved to pickle files (this is useful if you don't want to re-run the dataset reader each time)
- `nabert-attn.json` contains a config for the basic NABERT model showing how to set a number representation different from the default first tag representation
- `nabert-plus.json` contains a config for NABERT+ with only standard numbers, where the standard numbers in the example are 100 and 1
- `nabert-plus-templated.json` contains a config for NABERT+ with standard numbers and templates, where the standard numbers in the example are 100 and 1
