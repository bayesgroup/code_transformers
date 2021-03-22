# Downloading and resplitting data

All scripts should be run from `data_utils` directory: `cd data_utils`.

For Python150k dataset:
```(bash)
bash get_data.sh py ../data
python resplit_data.py --base_dir ../data --dataset py --redistributable
```

For JavaScript150k dataset:
```(bash)
bash get_data.sh js ../data
python resplit_data.py --base_dir ../data --dataset js
```

The scripts create `data` folder in the root directory of the repository, download data into ths folder and resplit data. If you have already downloaded and unpacked data from official Py/JS150k website, skip `get_data.sh` step and specify your path to the unpacked archive in `--base_dir` flag.

### Directory structure
* `get_data.sh`: downloads data from https://www.sri.inf.ethz.ch/
* `resplit_data.py`: creates new train / val / test `.json` and `.txt` files (with splitting by repository, removing duplicates and the redistributable version of Python150k)
* `py_redistributable_repos`: lists of repositories that are redistributable (for Python150k) from https://github.com/google-research-datasets/eth_py150_open
* `duplicates`: lists of duplicating examples from https://ieee-dataport.org/open-access/deduplication-index-big-code-datasets
* `repos`: lists of repositories for train / val / test

### Data statisics

We actually split data by GitHub usernames, not repositories, i. e. put all code of one username to one of train / val / test partitions.

For Python150k dataset:

|Partition | # GitHub usernames | # examples (code files) |
|-------------|---------:|-------------:|
|    Train    |  3303  |  76467 |
|    Val    |  367  |  8004  |
|    Test    |  1925  |  38694  |


For JavaScript150k dataset:

|Partition | # GitHub usernames | # examples (code files) |
|-------------|---------:|-------------:|
|    Train    |  5770  |  69038  |
|    Val    |  641  |  8665  |
|    Test    |  3206  |  41542  |
