#!/bin/bash
# Source this file to activate the flaxdiff conda environment:
#   source activate_env.sh
export PATH=$HOME/miniconda3/envs/flaxdiff/bin:$HOME/miniconda3/bin:$PATH
export TOKENIZERS_PARALLELISM=false
echo "Activated flaxdiff environment ($(python --version))"
