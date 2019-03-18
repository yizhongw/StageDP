# Two-stage Discourse Parser

Here we implement the RST discourse parser described in [A Two-stage Parsing Method for Text-level Discourse Analysis](http://aclanthology.coli.uni-saarland.de/pdf/P/P17/P17-2029.pdf). 

The best-trained models are put in the `data/model` folder. Due to the licence of RST data corpus, we can't include the data in our project folder. To reproduce the result in the paper, you need to download it from the LDC, preprocess the data as we state below and evaluate the model with `python3 main.py --eval --eval_dir EVAL_DIR`.  

### Usage:

1. Preprocess the data:

    
    This project relies on Stanford CoreNLP toolkit to preprocess the data. You can download from [here](http://stanfordnlp.github.io/CoreNLP/index.html) and put the file [run_corenlp.sh](./run_corenlp.sh) into the CoreNLP folder. Then use the following command to preprocess both the data in train_dir and in test_dir.
    
    ```
    python3 preprocess.py [-h] --data_dir DATA_DIR --corenlp_dir CORENLP_DIR
    ```

2. Extract feature templates, action maps and relation maps:

    ```
    python3 main.py --prepare --train_dir TRAIN_DIR --eval_dir EVAL_DIR
    ```

3. Train model:

    ```
    python3 main.py --train --train_dir TRAIN_DIR
    ```
    
4. Evaluate model:

    ```
    python3 main.py --eval --eval_dir EVAL_DIR
    ```

### Requirements:

All the codes are tested under Python 3.5. And see requirements.txt for python library dependency and you can install them with pip.


