#! /bin/bash
# the script for Code completion data preprocessing
# creates data points for 3 setups:
# Struct+Text (struct_names dir)
# Struct (struct dir)
# Text (names dir)

set -e
PYTHON=`/usr/bin/env python3`

for data_type in py js
do

    path=../../data # ast dataset
    base_dir=../data/processed_data_"$data_type"/ # directory where to store processed dataset
    train="train_$data_type.dedup" # deduplicated asts
    test="test_$data_type.dedup"
    eval="val_$data_type.dedup"
    # load_data $data_type $path

    if [[ $data_type = "js" ]]; then
        # remove 0 at the end of ast
        # unify with py
        $PYTHON ./src/scripts/preprocess_js.py --iast $path/$train.json
        $PYTHON ./src/scripts/preprocess_js.py --iast $path/$test.json
        $PYTHON ./src/scripts/preprocess_js.py --iast $path/$eval.json
    fi

    struct_names_dir="struct_names" # A: types and values
    struct_dir="struct" # B: types and anonimized values
    names_dir="names" # C: values only
    n_context="500" # maximum size of the ast tree
    from_scratch="True" # if generate all data from scratch
    root_paths="True" # if use relative tree encodings
    tree_rel="True" # if use tree relative masks
    anon_vocab="500" # size of anonimized vocabulary

    function create_vocab () {
        dps_path=$1
        output_dir=$2
        mode=$3
        echo "---CREATE VOCAB----"
        for type in "values" "types"
        do
            if [[ $mode = "values" ]] && [[ $type = "types" ]]; then
                continue
            fi
            out=$output_dir/train/vocab.$type.pkl
            if [[ ! -f $out ]] || [[ $from_scratch = "True" ]]; then
                echo "Creating vocab..."
                $PYTHON ./src/scripts/generate_vocab.py -i $dps_path -o $out -t $type
                echo "Created vocab at $out."
                ln -sfn ../train/vocab.$type.pkl $output_dir/eval/vocab.$type.pkl
                ln -sfn ../train/vocab.$type.pkl $output_dir/test/vocab.$type.pkl
            else
                echo "Vocab is already created at $out Skip."
            fi
        done
    }

    function create_tree_relative_mask_vocab () {
        output_dir=$1
        if [ $tree_rel = "True" ]; then
            echo "---CREATE TREE RELATIVE MASK VOCAB----"
            out=$output_dir/train/rel_vocab.pkl
            if [ ! -f $out ] || [ $from_scratch = "True" ]; then
                echo "Creating rel vocab..."
                $PYTHON ./src/scripts/generate_vocab.py -t rel -i $output_dir/train/tree_rel.txt -o $out
                echo "Created vocab at $out."
                ln -sfn ../train/rel_vocab.pkl $output_dir/eval/rel_vocab.pkl
                ln -sfn ../train/rel_vocab.pkl $output_dir/test/rel_vocab.pkl
            else
                echo "Vocab is already created at $out Skip."
            fi
        fi
    }

    function generate_tree_masks () {
        output_dir=$1
        if [ ! $tree_rel = "True" ]; then
            return
        fi
        echo "---GENERATE TREE MASKS----"
        if [ ! -f "$output_dir/train/tree_rel.txt" ] || [ $from_scratch = "True" ]; then
            echo "Creating train/eval tree masks... May take a while!"
            $PYTHON ./src/scripts/generate_tree_masks.py --iast $path/$train.json --out $output_dir/train/tree_rel.txt --n_ctx $n_context
            $PYTHON ./src/scripts/generate_tree_masks.py --iast $path/$eval.json  --out $output_dir/eval/tree_rel.txt  --n_ctx $n_context
            $PYTHON ./src/scripts/generate_tree_masks.py --iast $path/$test.json  --out $output_dir/test/tree_rel.txt  --n_ctx $n_context    
            echo "Created tree masks at $output_dir"
        else
            echo "Train/eval tree masks are already created. Skip."
        fi
        create_tree_relative_mask_vocab $output_dir
    }

    function create_tree_relative_encodings () {
        output_dir=$1
        if [ $root_paths = "True" ]; then
            echo "---CREATE TREE RELATIVE ENCODINGS----"
            if [ ! -f $out ] || [ $from_scratch = "True" ]; then
                echo "Creating root paths..."
                $PYTHON ./src/scripts/generate_root_paths.py --ast_fp $path/$train.json -o $output_dir/train/paths.txt --n_ctx $n_context
                $PYTHON ./src/scripts/generate_root_paths.py --ast_fp $path/$eval.json  -o $output_dir/eval/paths.txt  --n_ctx $n_context
                $PYTHON ./src/scripts/generate_root_paths.py --ast_fp $path/$test.json  -o $output_dir/test/paths.txt  --n_ctx $n_context
                echo "Created eval root paths at $output_dir"
            else
                echo "Root paths are already created at $out Skip."
            fi
        fi
    }

    function generate_data_points () {
        output_dir=$1
        mode=$2
        echo "---GENERATE DATA POINTS----"
        if [ ! -f "$output_dir/train/dps.txt" ] || [ $from_scratch = "True" ]; then
            echo "Creating train/eval datapoints... May take a while!"
            $PYTHON ./src/scripts/generate_data.py --iast $path/$train.json --oast $output_dir/train/dps.txt --itxt $path/$train.txt --otxt $output_dir/train/files.txt --n_ctx $n_context --mode $mode
            $PYTHON ./src/scripts/generate_data.py --iast $path/$eval.json  --oast $output_dir/eval/dps.txt  --itxt $path/$eval.txt  --otxt $output_dir/eval/files.txt  --n_ctx $n_context --mode $mode
            $PYTHON ./src/scripts/generate_data.py --iast $path/$test.json  --oast $output_dir/test/dps.txt  --itxt $path/$test.txt  --otxt $output_dir/test/files.txt  --n_ctx $n_context --mode $mode
            echo "Created datapoints at $output_dir"
        else
            echo "Train/eval datapoints are already created. Skip."
        fi
        create_vocab $output_dir/train/dps.txt $output_dir $mode
        if [ $mode = "all" ]; then
            generate_tree_masks $output_dir
            create_tree_relative_encodings $output_dir $mode
        fi
    }

    function generate_anon_data_points () {
        struct_names_dir=$1
        output_dir=$2
        echo "---GENERATE ANONIMIZED DATA POINTS----"
        if [ ! -f "$output_dir/train/dps.txt" ] || [ $from_scratch = "True" ]; then
            echo "Creating train/eval anonimized datapoints... May take a while!"
            for mod in "train" "test" "eval"; do
                $PYTHON ./src/scripts/generate_anon_data.py --in_fp $struct_names_dir/$mod/dps.txt --out_anon $output_dir/$mod/dps.txt --vocab_size $anon_vocab
                if [ $tree_rel = "True" ]; then
                    cp $struct_names_dir/$mod/tree_rel.txt $output_dir/$mod/
                    cp $struct_names_dir/$mod/rel_vocab.pkl $output_dir/$mod/
                fi
            done
            echo "Created datapoints at $output_dir"
            create_vocab $output_dir/train/dps.txt $output_dir
            create_tree_relative_encodings $output_dir
        else
            echo "Train/eval datapoints are already created. Skip."
        fi
    }

    function prepare_dir () {
        # rm -rf $1
        mkdir -p $1
        mkdir -p $1/train
        mkdir -p $1/eval
        mkdir -p $1/test
    }

    function set_struct_names () {
        output_dir=$base_dir/$struct_names_dir
        prepare_dir $output_dir
        generate_data_points $output_dir all
    }

    function set_struct () {
        input_dir=$base_dir/$struct_names_dir
        output_dir=$base_dir/$struct_dir
        prepare_dir $output_dir
        generate_anon_data_points $input_dir $output_dir
    }

    function set_names () {
        output_dir=$base_dir/$names_dir
        prepare_dir $output_dir
        generate_data_points $output_dir values
    }
        

    function main () {
        echo "preprocessing"
        set_struct_names
        set_struct
        set_names
    }

    main
done