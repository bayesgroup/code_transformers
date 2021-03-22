# Comparation of Struct methods
args=${@:1}
for lang in "py" "js"
do
    project="code_transformers"
    run_group="experiment-2a-$lang"
    set -e

    # all experiments with positional embeddings
    common_args="--project $project --run_group $run_group --seed 1111"
    data_dir=../processed_data_$lang/


    # hyperparameters tuning:  
    # bash src/srun.sh --name $dataset_type.tree_pos_enc.8.4 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
    # --max_depth 8 --max_width 4
    # bash src/srun.sh --name $dataset_type.tree_pos_enc.32.16 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
    # --max_depth 32 --max_width 16
    # bash src/srun.sh --name $dataset_type.tree_pos_enc.64.2 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
    # --max_depth 64 --max_width 2
    # bash src/srun.sh --name $dataset_type.tree_pos_enc.8.16 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
    # --max_depth 8 --max_width 16 
    # bash src/srun.sh --name $dataset_type.tree_pos_enc.16.8 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
    # --max_depth 16 --max_width 8

    # for vocab in 10000 1000 100 10  
    # do
    #     bash src/srun.sh \
    #         --name $dataset_type.tree_rel_att_add.$vocab \
    #         --use_tree \
    #         --additive \
    #         --tree_rel_max_vocab $vocab \
    #         --base_dir $data_dir/$dataset_type \
    #         $common_args
    # done

    # for rel_kmax in 8 32 128 250
    # do
    #     bash src/srun.sh \
    #         --name $dataset_type.seq_rel_att.$rel_kmax \
    #         --use_seq \
    #         --rel_kmax $rel_kmax \
    #         --base_dir $data_dir/$dataset_type \
    #         $common_args
    # done

    for dataset_type in struct_names struct
    do
        # 1 Sequential positional ebeddings
        bash src/srun.sh \
            --name $dataset_type.seq_pos_ebeddings \
            --use_pos_embed \
            --base_dir $data_dir/$dataset_type \
            $args \
            $common_args

        # 2 Sequential relative attention
        for rel_kmax in 32
        do
            bash src/srun.sh \
                --name $dataset_type.seq_rel_att.$rel_kmax \
                --use_seq \
                --rel_kmax $rel_kmax \
                --base_dir $data_dir/$dataset_type \
                $args \
                $common_args
        done

        # 3 Tree positional encodings
        if [[ $lang = "js" ]]; then
            bash src/srun.sh --name $dataset_type.tree_pos_enc.32.16 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
            --max_depth 32 --max_width 16 $args
        fi
        if [[ $lang = "py" ]]; then
            bash src/srun.sh --name $dataset_type.tree_pos_enc.8.16 --tree_pos_enc --base_dir $data_dir/$dataset_type $common_args \
            --max_depth 8 --max_width 16 $args
        fi

        # 4 Tree relative attention
        for vocab in 1000 
        do
            bash src/srun.sh \
                --name $dataset_type.tree_rel_att_add.$vocab \
                --use_tree \
                --additive \
                --tree_rel_max_vocab $vocab \
                --base_dir $data_dir/$dataset_type \
                $args \
                $common_args
        done
    done

    # echo "to view results run:"
    # echo "tensorboard dev upload --logdir GPT-CODE/$project/$run_group --name $project/$run_group"
done