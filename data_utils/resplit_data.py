import argparse
import json
import os

def get_repo(s):
    s = s[5:]
    return s[:s.find("/")]

class Python150kDataset:
    train_fn="python100k_train"
    eval_fn="python50k_eval"
    duplicates_fn="duplicates/py150-duplicates.json"
    lang="py"
    dupfilter = lambda path: path[32:]

class JavaScript150kDataset:
    train_fn="programs_training"
    eval_fn="programs_eval"
    duplicates_fn="duplicates/js150-duplicates.json"
    lang="js"
    dupfilter = lambda path: path

def get_avaliable_datasets():
    return {"py": Python150kDataset, "js": JavaScript150kDataset}

def filter_redistributable(dataset, args):
    assert dataset.lang == "py", "redistributable version not availiable for js"
    redistr_paths = set()
    paths = [ f"py_redistributable_repos/{mode}__manifest.json" \
                            for mode in ["dev", "train", "eval"] ]
    for path in paths:
        with open(path) as fr:
            man = json.load(fr)
        for f in man:
            redistr_paths.add(f["filepath"])

    def is_redistributable(path):
        return path[5:] in redistr_paths
    return is_redistributable
    
def read_repos(fn):
    with open(fn) as f:
        return f.read().split("\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess 150k dataset")
    parser.add_argument("--base_dir", type=str, default="../data/", help="path to the folder with json and txt files downloaded from www.sri.inf.ethz.ch")
    parser.add_argument("--dataset", type=str, choices=get_avaliable_datasets(), help="Dataset: py or js")
    parser.add_argument("--redistributable", action="store_true",
                        help="only for python")
    args = parser.parse_args()

    dataset = get_avaliable_datasets()[args.dataset]
    args.train = dataset.train_fn
    args.eval = dataset.eval_fn
    args.lang = dataset.lang
    args.duplicates = dataset.duplicates_fn

    is_redistributable = lambda path: True
    if args.redistributable:
        is_redistributable = filter_redistributable(dataset, args)

    json_train = os.path.join(args.base_dir, f"{args.train}.json")
    json_test = os.path.join(args.base_dir, f"{args.eval}.json")

    json_train_out = os.path.join(args.base_dir, f"train_{args.lang}.dedup.json")
    json_val_out = os.path.join(args.base_dir, f"val_{args.lang}.dedup.json")
    json_test_out = os.path.join(args.base_dir, f"test_{args.lang}.dedup.json")

    txt_train = os.path.join(args.base_dir, f"{args.train}.txt")
    txt_test = os.path.join(args.base_dir, f"{args.eval}.txt")

    txt_train_out = os.path.join(args.base_dir, f"train_{args.lang}.dedup.txt")
    txt_val_out = os.path.join(args.base_dir, f"val_{args.lang}.dedup.txt")
    txt_test_out = os.path.join(args.base_dir, f"test_{args.lang}.dedup.txt")

    train_repos = read_repos("repos/train_repos_%s.txt"%args.lang)
    val_repos = read_repos("repos/val_repos_%s.txt"%args.lang)
    test_repos = read_repos("repos/test_repos_%s.txt"%args.lang)

    with open(args.duplicates, "r") as fin:
        dup_list = json.loads(fin.read())
    dups = set()
    for dup_elem in dup_list:
        for elem in dup_elem[1:]:
            dups.add(elem[27:] if args.lang=="py" else elem)

    num_train = 0
    num_val = 0
    num_test = 0
    dup_num = 0
    nowhere = 0
    
    seen = set()
    seen_count_infile = 0
    seen_count_notinfile = 0
    
    with open(json_train_out, "w") as fout_train, \
         open(json_val_out, "w") as fout_val,\
         open(json_test_out, "w") as fout_test, \
         open(txt_train_out, "w") as ftxtout_train, \
         open(txt_val_out, "w") as ftxtout_val, \
         open(txt_test_out, "w") as ftxtout_test:
        for json_file, txt_file in [[json_train, txt_train], [json_test, txt_test]]:
            with open(json_file, encoding='latin-1') as lines, open(txt_file) as fin:
                for line_index, (line, filename) in \
                                    enumerate(zip(lines, fin)):
                    if line_index % 10000 == 0:
                        print('Processing line: {}'.format(line_index))
                    filename = filename.strip()
                    repo = get_repo(filename)
                    if args.lang == "js":
                        filename = filename[5:]
                    line = line.strip()
                    
                    if line in seen:
                        if filename in dups:
                            seen_count_infile += 1
                        else:
                            seen_count_notinfile += 1
                    if not filename in dups and not line in seen and is_redistributable(filename):
                        if repo in train_repos:
                            fout_train.write(line)
                            fout_train.write("\n")
                            ftxtout_train.write(filename)
                            ftxtout_train.write("\n")
                            num_train += 1
                            seen.add(line)
                        elif repo in val_repos:
                            fout_val.write(line)
                            fout_val.write("\n")
                            ftxtout_val.write(filename)
                            ftxtout_val.write("\n")
                            num_val += 1
                            seen.add(line)
                        elif repo in test_repos:
                            fout_test.write(line)
                            fout_test.write("\n")
                            ftxtout_test.write(filename)
                            ftxtout_test.write("\n")
                            num_test += 1
                            seen.add(line)
                        else:
                            nowhere += 1
                    else:
                        dup_num += 1
    print(f"train files: {num_train}, val files: {num_val}, test files: {num_test}")

if __name__ == "__main__":
    main()
