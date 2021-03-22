# S | S+T | T
args=${@:1}
for lang in "py" "js"
do
    project="code_transformers"
    run_group="experiment-1-$lang"
    set -e

    # all experiments with positional embeddings
    common_args="--project $project --run_group $run_group --seed 1111"
    data_dir=../processed_data_$lang/

    # sequential positional embedding or no positional information (ablation)
    for relative in "--use_pos_embed" ""
    do  
        bash src/srun.sh \
            --name struct_and_names.$relative \
            --base_dir $data_dir/struct_names \
            $relative \
            --use_test \
            $args \
            $common_args

        bash src/srun.sh \
            --name struct.$relative \
            --base_dir $data_dir/struct \
            $relative \
            --use_test \
            $args \
            $common_args

        sbatch -c 4 --gpus 1 src/srun.sh \
            --name names.$relative \
            --base_dir $data_dir/names \
            --only_values \
            $relative \
            --use_test \
            $args \
            $common_args
    done
done

# echo "to view results run:"
# echo "tensorboard dev upload --logdir GPT-CODE/$project/$run_group --name $project/$run_group"