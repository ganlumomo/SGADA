python core/pseudo_label.py \
 --trained ./sgada_model_files/sgada_domain_warmup/best_model_12.pt \
 --d_trained ./sgada_model_files/sgada_domain_warmup/d_best_model_12.pt \
 --lr 1e-5 --d_lr 1e-3 --batch_size 1 \
 --lam 0.25 --thr 0.79 --thr_domain 0.87 \
 --device cuda:1
