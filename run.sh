set -o nounset
set -o errexit
set -o pipefail

{
    args=$@
    for arg in $args; do
        eval "$arg"
    done

    echo "n_observed_chars:      ${n_observed_chars:=8}"
    echo "n_beam:                ${n_beam:=12}"
    echo "batch_size:            ${batch_size:=200}"
    echo "alpha:                 ${alpha:=0.4}"
    echo "beta:                  ${beta:=0.5}"
    echo "model:                 ${model:=Baichuan2-7B-Base}"
    echo "bert:                  ${bert:=relm}"
    echo "base_suite:            ${base_suite:=relm_trained_on_sighans}"

    suite="${base_suite}.${alpha}_${beta}.n_beam-${n_beam}.n_observed_chars-${n_observed_chars}"

    datasets=(
        # "ecspell/law.test.para"
        # "ecspell/med.test.para"
        # "ecspell/odw.test.para"
    )

    for dataset in "${datasets[@]}"; do
        dataset_name=$(echo "${dataset}" | cut -d. -f1)
        dataset_name=${dataset_name//\//_}
        mkdir -p "results/${model}/${suite}/${dataset_name}"
        python -u run.py \
            --input-file "datasets/${dataset}"  \
            --path "results/${model}/${suite}/${dataset_name}"  \
            --model-name "${model}"  \
            --bert-name "${bert}"  \
            --n-observed-chars "${n_observed_chars}"  \
            --n-beam "${n_beam}"  \
            --batch-size "${batch_size}"  \
            --alpha "${alpha}"  \
            --beta "${beta}"  \
            --use-faithfulness-reward | tee "results/${model}/${suite}/${dataset_name}/prediction.log"
        python eval/evaluate.py \
            --gold "datasets/${dataset}"  \
            --hypo "results/${model}/${suite}/${dataset_name}/prediction.txt"  \
            --to_halfwidth  \
            --ignore_unmatch_length  \
            --ignore_space  \
            --ignore_punct  \
            --ignore_letter \
    done

}
