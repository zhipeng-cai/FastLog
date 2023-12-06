# **Artifact for the Paper "FastLog: An End-to-End Method to Efficiently Generate and Insert Logging Statements"**

This is the supplementary material for the paper "*FastLog: An End-to-End Method to Efficiently Generate and Insert Logging Statements*".

It contains:

1. **The complete experimental results for this paper:** the complete experimental results on both the original dataset and the new dataset for all research questions.
2. **The datasets used in our experiments:** the original dataset used in [LANCE](https://github.com/antonio-mastropaolo/LANCE) and the new dataset constructed by ourselves.
3. **The outputs of our experiments:** the released model checkpoints and prediction results.
4. **The replication package for our experiments:** the scripts to replicate our experiments.

------

## Complete Experimental Results

Due to the space limit for the paper, we have not included all the detailed experimental results (the results on the original datasets of RQ3 and RQ4) in our paper. Here we provide the complete experimental results on both the original dataset and the new dataset of all RQs in the directory `/complete-results`.



## Datasets

We have released both the original dataset used in  [LANCE](https://github.com/antonio-mastropaolo/LANCE)  and the new dataset constructed by ourselves in our experiments. Specifically, we provide the raw format datasets (the Java method text), and the processed format datasets (specific input and output format for the models in Stage-1 and Stage-2). The datasets can be downloaded from [here](https://drive.google.com/drive/folders/1mc7SPETNUEfSeAOtH6o7D8xsVsC5Vx3O?usp=sharing) and put them under the directory `/datasets`.



## Outputs

We have released the model checkpoints and the prediction results in our experiments. Specifically, we provide two model checkpoints for Stage-1 (logging position prediction) and Stage-2 (logging statement generation) in our methodology. Besides, we also provide the prediction results of Stage-1 and Stage-2 and the final prediction results on both the original and the new datasets. All models and prediction results can be downloaded from [here](https://drive.google.com/drive/folders/1ai1FLGveoIOMi2QDvnsgRhnQjVmWR_XJ?usp=sharing) and put them under the directory `/output`.

### File Structure

```
output
├── models
│   ├── stage1  # the model checkpoint for Stage-1 to predict logging positions
│   └── stage2  # the model checkpoint for Stage-2 to generate logging statements
└── predictions
    ├── new-dataset
    │   ├── positions.csv                 # the prediction results of Stage-1
    │   ├── statements_beam_search.csv    # the prediction results of Stage-2 using beam search
    │   ├── statements_greedy_search.csv  # the prediction results of Stage-2 using greedy search
    │   ├── results_beam_search.csv       # the final prediction results using beam search
    │   └── results_greedy_search.csv     # the final prediction results using greedy search
    └── original-dataset
        ├── positions.csv
        ├── statements_beam_search.csv
        ├── statements_greedy_search.csv
        ├── results_beam_search.csv
        └── results_greedy_search.csv
```



## Replication Package

We have released the relevant code to replicate our experiments.


### File Structure

```
src
├── eval
│   ├── bleu_calculator.py             # calculate the BLEU metric
│   ├── rouge_calculator.py            # calculate the ROUGE metric
│   └── run_eval.py                    # run to evaluate the prediction results
├── metrics
│   └── sequval.py                     # the util script to calculate sequval metirc for selecting the best model
├── get_multiple_position.py           # get multiple prediciton results for logging positions
├── get_multiple_statement.py          # get multiple prediciton results for logging statements
├── insert_log.py                      # insert the generated logging statements
├── plbart_for_tokenclassification.py  # the util script to use PLBART model for token-classification
├── stage1_predict.py                  # predict logging positions in Stage-1
├── stage1_trainer.py                  # train the model in Stage-1
├── stage2_predict.py                  # generate logging statements in Stage-2
└── stage2_trainer.py                  # train the model in Stage-2
```

### Replication Instruction

0. Install the necessary Python 3.7 libraries with `requirements.txt`.

   ```
   pip install -r requirements.txt
   ```

1. Run `src/stage1_trainer.py` and `src/stage2_trainer.py` to fine-tune the PLBART model for Stage-1 to predict logging positions and for Stage-2 to generate logging statements, respectively.  You can specify the corresponding parameters at the beginning of the scripts, such as the `learning_rate` and `batch_size`.

2. Run `src/stage1_predict.py` to predict the logging positions.

3. Run `src/stage2_predict.py` to generate the logging statements for the predicted logging positions obtained in Stage-1.

4. Run `src/insert_log.py` to insert the generated logging statements into the corresponding predicted logging positions.

5. (Optional) Run `src/get_multiple_position.py` and  `src/get_multiple_statement.py` to get multiple predicted logging positions and corresponding multiple logging statements (as mentioned in the Discussion Section in our paper).

6. Run `src/eval/run_eval.py` to calculate the Accuracy, BLEU, and ROUGE metrics for the prediction results.