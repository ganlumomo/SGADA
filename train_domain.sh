python core/domain.py \
 --trained ./sgada_model_files/sgada_source_label.pt \
 --lr 1e-5 --d_lr 1e-3 --g_lr 1e-3 --batch_size 1 \
 --lam 0.25 --lam_NCE 1.0 --thr 0.79 --thr_domain 0.87 \
 --device cuda:0
