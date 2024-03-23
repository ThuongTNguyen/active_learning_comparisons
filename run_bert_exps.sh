CUDA_VISIBLE_DEVICES=7 python bert_exps.py --dsname pubmed --bs 200 --trial 1 --qm random > log_pubmed_200_random  &
CUDA_VISIBLE_DEVICES=6 python bert_exps.py --dsname pubmed --bs 200 --trial 1 --qm real > log_pubmed_200_real &
CUDA_VISIBLE_DEVICES=5 python bert_exps.py --dsname pubmed --bs 200 --trial 1 --qm cal > log_pubmed_200_cal &
CUDA_VISIBLE_DEVICES=2 python bert_exps.py --dsname pubmed --bs 200 --trial 1 --qm dal > log_pubmed_200_dal &
CUDA_VISIBLE_DEVICES=1 python bert_exps.py --dsname pubmed --bs 200 --trial 1 --qm margin > log_pubmed_200_margin &

#CUDA_VISIBLE_DEVICES=4 python bert_exps.py --dsname pubmed --bs 500 --trial 1 --qm random --qm real --qm cal --qm margin --qm dal &
