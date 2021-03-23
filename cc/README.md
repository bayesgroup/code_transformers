# Code Completion Task

## Dependencies
The implementation uses the following libraries:
* torch (we used version 1.5.1, you can use versions >=1.5 and it should work, but we did not test the code on other versions)
* pytorch-lightning (we used version 0.8.5)
* numpy
* tqdm

You can optionally run `pip install -r requirements.txt` to install all dependencies (using a virtual environment is highly recommended to avoid downgrading libraries to earlier versions).

## Running experiments
1. Download and resplit data, see `../data_utils` for details;
2. Preprocess data for the CC task, see section "Data preprocessing" below;
3. Run the experiment you are interested in, see section "Training models" below.

All scripts below should be run from `cc/main` directory: `cd cc/main`.

## Data Preprocessing
From `main` directory run `bash src/scripts/preprocess_data.sh`. 

The script creates a data directory `.../data/processed_data_{py,js}/` with processed asts from an input directory `../../data/` with raw asts. 
The script creates `struct_names`, `struct`, `names` folders. These are for `Syntax+Text`, `Syntax`, `Text` models from the empirical study paper. Each folder contains `train`, `eval`, `test` subfolder with the following files:

* `dps.txt`: types, values, memory length
* `paths.txt`: paths from the root (for Tree positional encodings)
* `tree_rel.txt`: data for Tree relative attention

## Training models
In case you only wish to train a __baseline__ for your work (and not to repeat our experiments), use the sequential relative attention model.

To train a model, e.g. with sequential relative attention, use `train.py` (select `py` or `js`):

```(bash)
python3 train.py --name baseline --work_dir exp_dir --project project_name --base_dir ../data/processed_data_{py,js}/struct_names --num_workers 4 --gpus 1 --use_seq | tee log.txt
```

We also suggest passing flag `--use_anonymized` to the `train.py`, in order to use our [anonymization of the out-of-vocabulary identifiers](https://arxiv.org/abs/2010.12663). This simple technique will increase the test quality by several percent.

All hyperparameters are given in the default options of arguments in `train.py` and `model.py` scripts. The best model checkpoint will be saved in `exp_dir/project_name/`. The validation performance is logged after each training epoch. Test performance is logged for the best model after the training. To log on the `test` partition instead of `val` at the end of an epoch, add `--use_test` flag.
    
For different models from the empirical study paper, pass:
1. `--use_pos_embed` for Sequential positional ebeddings
2. `--use_seq` for Sequential relative attention (use flag `--rel_kmax <rel_kmax>` to change the hyperparameter) 
3. `--tree_pos_enc` for Tree positional encodings (use flags `--max_depth <max_depth> --max_width <max_width>` to change the hyperparameters)
4. `--use_tree --additive` for Tree relative attention (use flags `--tree_rel_max_vocab <max vocab>` to change the hyperparameters, remove `--additive` to use the multiplicative version from Kim et al., 2020)

For different data sources from the empirical study paper, specify the following folder in the `--base_dir` argument:
1. `struct_names`: Syntax+Text
2. `struct`: Syntax
3. `names`: Text. Also needs flag `--only_values`. We trained this model on 2 Tesla V100 32 GB `--gpus 2`.

You can also run `train.py` script in the test-only mode by passing flag `--eval` and specifying path to the model in the `--restart_cpt <path to .ckpt>` flag.

## Reproducing the experiments from the empirical study paper
    
Consider the following _optional_ preliminary steps:
* run `bash experiments/preprocess.sh` to convert all the data to the internal format. You need to do it if you want to run experiments on the same data type in parallel;
* configure `src/srun.sh`, e.g. set the number of GPUs/CPUs, or pass `--gpus 1 --num_workers 4` directly to the training script. 
    
To repeat the experiments from the paper, run `bash experiments/exp{1,2a,2b}.sh`:
* `exp1.sh` trains the `Syntax+Text`, `Syntax`, `Text` models with positional embeddings or without any positional encodings (Figure 2 and Table 1 from the paper). 
* `exp2a.sh` trains the `Syntax+Text`, `Syntax` models with 1) Sequential positional ebeddings, 2) Sequential relative attention 3) Tree positional encodings 4) Tree relative attention. (Figure 4 from the paper). In this script, you can also find code for hyperparameter tuning.
* `exp2b.sh` trains the `Syntax+Text` model with a combination of Sequential relative attention and other techniques (Table 2 from the paper).

For ensembling, see `main/src/scripts/ensemble.py`.

The instructions for reproducing the experients from the OOV anonymization paper will be released soon.

## Attribution

The code was partly borrowed from [this repo](https://github.com/facebookresearch/code-prediction-transformer).
    
