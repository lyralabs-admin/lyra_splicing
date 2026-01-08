# Lyra Splice Classifier

This project contains code for training and evaluating the Lyra splice site classifier, a bidirectional model designed for high-accuracy splice site prediction.

## Setup

1.  **Create and set up the virtual environment**:
    Run the provided setup script. This will create a virtual environment named `lyra_env` and install all the required dependencies. The code currently requires FlashFFTCConv, which needs to be run on H100/A100. 
    ```bash
    bash setup_venv.sh
    ```

2.  **Activate the environment**:
    Before running any scripts, activate the virtual environment:
    ```bash
    source lyra_env/bin/activate
    ```

## Dataset

The training and test data was generated using the `create-data` command from the OpenSpliceAI repository [[https://github.com/Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI)]. Processed data can be downloaded from hugging face at nmjaved/openspliceai_data_human_10k and placed in the folder openspliceai_data_human_10k

The following command was used to create the dataset from the GRCh38 genome assembly and MANE v1.3 annotations, defining a 10,000 bp flank around each core region with default settings for paralogue removal and data split definition:

```bash
openspliceai create-data \
   --genome-fasta  GCF_000001405.40_GRCh38.p14_genomic.fna \
   --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff \
   --output-dir train_test_dataset_MANE/ \
   --flanking_size 10000
```

## Best Model & Performance

The best performing model, saved as `dm48_ds48_22blocks.pt`, was trained using the default parameters in the `train_lyra_splice_classifier.py` script.

-   **Model Parameters**:
    -   `d_model`: 48
    -   `d_state`: 48
    -   `num_blocks`: 22
-   **Total Parameters**: 714,243

### Test set performance metrics (10k context)

| Model                       | Acceptor Top-k Acc | Donor Top-k Acc | Avg Top-k Acc |
| --------------------------- | ------------------ | --------------- | ------------- |
| SpliceAI / OpenSpliceAI     | 0.9315             | 0.9312          | 0.9314        |
| **Lyra (This Repository)**  | **0.9417**         | **0.9367**      | **0.9392**    |


## How to Run
Commands should be run from within the `lyra_splicing/` directory.

### Training

The training script allows for tuning of model and optimizer hyperparameters.

**Default Training Command:**
```bash
torchrun --nproc_per_node=2 train_lyra_splice_classifier.py \
    --ddp \
    --train_h5 openspliceai_data_human_10k//dataset_train.h5 \
    --test_h5 openspliceai_data_human_10k//dataset_test.h5 \
    --out_dir lyra_runs/default_model
```


### Evaluation

When evaluating a model, you must provide the same architecture parameters (`d_model`, `d_state`, `num_blocks`) that were used for training.

**Example Evaluation Command for the best model:**
```bash
python evaluate_model.py \
    --model_path dm48_ds48_22blocks.pt \
    --test_h5 openspliceai_data_human_10k//dataset_test.h5 \
    --d_model 48 \
    --d_state 48 \
    --num_blocks 22
```

### Interpretability
Coming soon...
