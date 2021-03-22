#!/usr/bin/bash
set -e

bash experiments/preprocess.sh # convert to vocab

bash experiments/exp1.sh ${@:1}

bash experiments/exp2a.sh ${@:1}

bash experiments/exp2b.sh ${@:1}