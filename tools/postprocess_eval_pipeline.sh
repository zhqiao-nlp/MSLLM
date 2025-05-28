#python tools/postprocess.py \
#    --para "/public/home/zhli13/zhqiao/llm-csc-main/datasets/ecspell/test.med.para.cut62" \
#    --pred "/public/home/zhli13/zhqiao/llm-csc-main/results/public/home/zhli13/zhqiao/Baichuan2-7B-Base/v1.alpha-1.n_beam-12.n_observed_chars-8/ecspell_test/prediction.txt" \

python eval/evaluate.py \
            --gold "/public/home/zhli13/zhqiao/llm-csc-main/datasets/ecspell/test.med.para.cut62"  \
            --hypo "/public/home/zhli13/zhqiao/llm-csc-main/results/public/home/zhli13/zhqiao/Baichuan2-7B-Base/v1.alpha-1.n_beam-12.n_observed_chars-8/ecspell_test/prediction.txt"  \
            --to_halfwidth  \
            --ignore_unmatch_length  \
            --ignore_space  \
            --ignore_punct  \
            --ignore_letter