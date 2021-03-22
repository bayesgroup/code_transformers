project="code_transformers"
run_group="N/A"
set -e

args=${@:1}
for lang in "py" "js"
do
    common_args="--project $project --run_group $run_group --seed 1111"
    data_dir=../processed_data_$lang/

    for dataset_type in struct struct_names
    do
        for vocab in 1000 
        do
            bash src/srun.sh \
                --name N/A \
                --use_tree \
                --additive \
                --tree_rel_max_vocab $vocab \
                --base_dir $data_dir/$dataset_type \
                --preprocess \
                --debug \
                $args \
                $common_args
        done
    done

    for dataset_type in names
    do
        bash src/srun.sh \
            --name N/A \
            --base_dir $data_dir/$dataset_type \
            --preprocess \
            --only_values \
            --debug \
            $args \
            $common_args
    done
done