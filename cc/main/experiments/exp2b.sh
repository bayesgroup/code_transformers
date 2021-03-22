# Compositions of SRA + other
args=${@:1}
for lang in "py" "js"
do
    project="code_transformers"
    run_group="experiment-2b-$lang"
    set -e

    # all experiments with positional embeddings
    common_args="--project $project --run_group $run_group --seed 1111"
    data_dir=../processed_data_$lang/

    dataset_type="struct_names"

    # 1 Seq rel attn + seq pos emb
    bash src/srun.sh \
        --name $dataset_type.use_seq.use_pos_embed \
        --use_pos_embed \
        --use_seq \
        --rel_kmax 32 \
        $args \
        --base_dir $data_dir/$dataset_type  \
        $common_args

    # 2 Seq rel attn + tree pos enc
    if [[ $lang = "js" ]]; then
        param="--max_depth 32 --max_width 16"
    fi
    if [[ $lang = "py" ]]; then
        param="--max_depth 8 --max_width 16"
    fi
    bash src/srun.sh \
        --name $dataset_type.use_seq.tree_pos_enc \
        --tree_pos_enc \
        --use_seq \
        --rel_kmax 32 \
        $param \
        $args \
        --base_dir $data_dir/$dataset_type  \
        $common_args

    # 3 Seq rel attn + tree rel attn
    bash src/srun.sh \
        --name $dataset_type.use_seq.use_tree.additive \
        --use_seq \
        --rel_kmax 32 \
        --use_tree \
        --tree_rel_max_vocab 1000 \
        --additive \
        $args \
        --base_dir $data_dir/$dataset_type  \
        $common_args


    # tensorboard dev upload --logdir GPT-CODE/$project/$run_group  --name $project/$run_group
done