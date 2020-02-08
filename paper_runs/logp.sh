python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 0 --dropout_rate 0.1 \
 --lr 1e-4 -o saved_models/moses_logp_noattn.pt -i moses/train.smi --precomputed_images moses/train_images.npy \
 --precomputed_values moses/train_logp.npy

python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 0 --dropout_rate 0.1 \
  --lr 1e-4 -o saved_models/moses_logp_noattn.pt -i moses/test.smi --precomputed_images moses/test_images.npy \
  --precomputed_values moses/test_logp.npy --eval_train  > moses_logp_performance.txt

python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 0 --dropout_rate 0.1 \
  --lr 1e-4 -o saved_models/moses_logp_noattn.pt -i moses/test_scaffold.smi --precomputed_images moses/test_scaffold_images.npy \
  --precomputed_values moses/test_scaffold_logp.npy --eval_train  >> moses_logp_performance.txt


#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
# --lr 1e-4 -o saved_models/moses_logp.pt -i moses/train.smi --precomputed_images moses/train_images.npy \
# --precomputed_values moses/train_logp.npy
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
#  --lr 1e-4 -o saved_models/moses_logp.pt -i moses/test.smi --precomputed_images moses/test_images.npy \
#  --precomputed_values moses/test_logp.npy --eval_train  > moses_logp_performance.txt
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
#  --lr 1e-4 -o saved_models/moses_logp.pt -i moses/test_scaffold.smi --precomputed_images moses/test_scaffold_images.npy \
#  --precomputed_values moses/test_scaffold_logp.npy --eval_train  >> moses_logp_performance.txt
#
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
# --lr 1e-4 -o saved_models/moses_logp_bw.pt -i moses/train.smi --precomputed_images moses/train_images.npy \
# --precomputed_values moses/train_logp.npy --bw
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
#  --lr 1e-4 -o saved_models/moses_logp_bw.pt -i moses/test.smi --precomputed_images moses/test_images.npy \
#  --precomputed_values moses/test_logp.npy --eval_train --bw  >> moses_logp_performance.txt
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
#  --lr 1e-4 -o saved_models/moses_logp_bw.pt -i moses/test_scaffold.smi --precomputed_images moses/test_scaffold_images.npy \
#  --precomputed_values moses/test_scaffold_logp.npy --eval_train --bw  >> moses_logp_performance.txt
#
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
# --lr 1e-4 -o saved_models/moses_logp_nopt.pt -i moses/train.smi --precomputed_images moses/train_images.npy \
# --precomputed_values moses/train_logp.npy --no_pretrain
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
#  --lr 1e-4 -o saved_models/moses_logp_nopt.pt -i moses/test.smi --precomputed_images moses/test_images.npy \
#  --precomputed_values moses/test_logp.npy --eval_train --no_pretrain  >> moses_logp_performance.txt
#
#python train.py -pb --rotate -p logp -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
#  --lr 1e-4 -o saved_models/moses_logp_nopt.pt -i moses/test_scaffold.smi --precomputed_images moses/test_scaffold_images.npy \
#  --precomputed_values moses/test_scaffold_logp.npy --eval_train --no_pretrain >> moses_logp_performance.txt