python train.py -pb --rotate -p logp -t 1 -b 512 --amp O2 -w 1 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
 --lr 1e-4 -o saved_models/moses_logp.pt -i moses/train.smi --precomputed_images moses/train_images.npy \
 --precomputed_values moses/train_logp.npy

python train.py -pb --rotate -p logp -t 1 -b 512 --amp O2 -w 1 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
  --lr 1e-4 -o saved_models/moses_logp.pt -i moses/test.smi --precomputed_images moses/test_images.npy \
  --precomputed_values moses/test_logp.npy --eval_train

python train.py -pb --rotate -p logp -t 1 -b 512 --amp O2 -w 1 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
  --lr 1e-4 -o saved_models/moses_logp.pt -i moses/test_scaffold.smi --precomputed_images moses/test_scaffold_images.npy \
  --precomputed_values moses/test_scaffold_logp.npy --eval_train