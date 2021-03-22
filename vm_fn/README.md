# Variable misuse and Function naming tasks

## Dependencies
The implementation uses the following libraries:
* torch (we used version 1.7.1, you can use versions >=1.5 and it should work, but we did not test the code on other versions)
* numpy (we used version 1.16.4)
* pandas
* tqdm
* prettytable
* tabulate
* nltk (imported only in FN task)
* psutil (imported only in FN task)

You can optionally run `pip install -r requirements.txt` to install all dependencies (using a virtual environment is highly recommended to avoid downgrading libraries to earlier versions).

## Running experiments
1. Download and resplit data, see `../data_utils` for details;
2. Preprocess data for a task you are interested in (VM, FN), see section "Data preprocessing" below;
3. Run the experiment you are interested in, see sections "Training baseline model" and "Reproducing experiments" below.

All scripts below should be run from `vm_fn` directory: `cd vm_fn`.

## Data preprocessing
Two steps: (1) basic preprocessing and (2) generating tree data (for tree-based Transformers).

### Step 1: basic preprocessing

#### Variable Misuse task (VM)
Run the following command (choose `py` or `js`):

```(bash)
python preprocess/select_functions_and_save_seq_data.py --lang {py|js} --task vm --base_dir ../data --output_dir preprocessed_data_vm
```

The script selects top-level functions, performs some filtering and saves data in `.json` format (for future preprocessing) and `.txt` files (to be passed to Transformer). The script is deterministic (except variable anonymization), and our targets (randomly generated synthetc bugs) are already stored in `preprocessed_data_vm` folder. In case you wish to generate targets youself, add `--generate_target True` flag (it will override our targets). Our preprocessing decisions are described in the docstrings inside the script.

#### Function Naming task (FN)

PY:

```(bash)
python preprocess/select_functions_and_save_seq_data.py --lang py --task fn --base_dir ../data --output_dir preprocessed_data_fn --generate_target True --tgt_vocab_size 15000
```

JS:

```(bash)
python preprocess/select_functions_and_save_seq_data.py --lang js --task fn --base_dir ../data --output_dir preprocessed_data_fn --generate_target True --tgt_vocab_size 7000
```

The script selects top-level functions, performs some filtering and saves data in `.json` format (for future preprocessing) and `.txt` files (to be passed to Transformer).  Our preprocessing decisions are described in the docstrings inside the script. Specifying `--tgt_vocab_size` is needed in functions filtering.

### Step 2: generating tree data

Run the following command (select `vm` or `fn`, `py`  or `js`, and any combination of remaining flags):

```(bash)
python preprocess/generate_tree_data.py --output_dir preprocessed_data_{vm|fn} --task {vm|fn} --lang {py|js} [--rel_matrices] [--root_paths] [--adj_edges]
```

Flag `--rel_matrices` generates additional data for tree relative attention (takes several hours), flag `--root_paths` --- for tree positional encodings (takes a few minutes), and flag `--adj_edges` --- for GGNN Sandwich (takes a few minutes).

## Training baseline model
In case you only wish to train a __baseline__ for your work, use the sequential relative attention model (select `vm` or `fn`, `py` or `js`):

```(bash)
python run_all.py --task {vm|fn} --lang {py|js} --exp_type exp2a_full --max_commands 1 [--test]
```

Use `--test` flag to check the generated command and remove flag to run the command. Make sure you have done basic data preprocessing (see above). To see commands for other AST-processing techniques, remove `--max_commands 1` flag.

We also suggest adding flag `--anonymize order` to the generated command for the VM task, in order to use our [anonymization of the out-of-vocabulary identifiers](https://arxiv.org/abs/2010.12663). This simple technique will increase the test quality by several percent.

## Reproducing the experiments from the empirical study paper

To train models, you can use `run_all.py` script. This script generates training commands for all experiments in the paper and stores hyperparameters for different model-dataset pairs.

Usage: 

```(bash)
python run_all.py --task {vm|fn} --lang {py|js} --exp_type {exp1|exp2a_full|exp2a_ano|exp2b_full} [--eval_part {test|val}] [--num_exps 1] [--test] [--label run] [--comment_add your_comment] [--tune_hypers]
```

Options:
* `--task` (required): Variable misuse (`vm`) or Function naming (`fn`)
* `--lang` (required): Python150k dataset (`py`) or JavaScript150k dataset (`js`). Make sure you have preprocessed data (see above), particularly, run `generate_tree_data.py` script in case you wish to use tree-based Transformers.
* `--exp_type` (required): what experiment to run, see details below
* `--eval_part`: which partition to evaluate on during training (options: `test`, `val`, default `test`)
* `--num_exps`: how many models of each kind to train (default: 1)
* `--test`: if specified, the commands will be only printed so you can check them; if not specified, the commands will be run
* `--label` and `--comment_add`: the logs and models are saved to a folder named `logs/{label}/{exp_group}/{exp_folder}{comment_add}`, and you can specify a general label for your current experiments (default label if `run`) and an additional comment for a partiular run (default empty)
* `--tune_hypers`: specify if you wish to tune hyperparameters; do not specify if you want to use predefined hyperparameters

Types of experiments (`--exp_type` option):
- `exp1`: Figure 2 and Table 3 in the paper (train vanilla Trasformer on input data of different kinds: parallel sequence of (type, value) pairs, only values or only types, parallel sequence with anonymized values etc.)
- `exp2a_full`: Figure 4 in the paper (train different Transformer modifications on parallel data without anonymization). If flag `--tune_hypers` is used, grid search is run, otherwise only training with optimal hyperparameters is performed
- `exp2a_ano`: Figure 5 in the Appendix (train different Transformer modifications on parallel data with anonymization)
- `exp2b_full`: Table 5 in the paper (combining sequential relative attetion with other Transformer modifications)

The `run_all.py` script generates commands like `python main/{vm|fn}/train.py ...`. To insert additional specifications into commands, e. g. use `sbatch`/`bsub` or set `CUDA_VISIBLE_DEVICES`, modify the last lines of the `run.py`. You can also specify additional training flags there, e. g. `--print_fq` (by default evaluation is performed after each epoch) or `--save_fq` (by default only the best model is saved).  

We also provide `run_eval.py` script that runs `train.py` in the test-only mode (you need to provide the paths to the models you wish to evaluate).

For ensembling, see scripts `main/{vm|fn}/ensembles.py`.

The instructions for reproducing the experients from the OOV anonymization paper will be released soon.

## Directory structure
* `preprocess`: scripts for preprocessing data for VM and FN tasks
* `trlib`: code for model parts, preprocessing etc. used in both tasks
* `main`: general scripts for constructing model, training model, ensembling
* `preprocessed_data_{vm|fn}`: data for VM and FN tasks generated during preprocessing will be saved in these folders

## Attribution
The code was partly borrowed from [this repo](https://github.com/wasiahmad/NeuralCodeSum).
