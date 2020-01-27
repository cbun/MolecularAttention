python train.py -pb --epochs 25 -i moses/train.smi --precomputed_values moses/train_sa.npy --lr 5e-5 -b 512 -w 2 -p sa \
  -t 1 --nheads 1 --dropout_rate 0.1 -o saved_models/moses_sa.pt --amp O2 --rotate --width 128 --depth 2  \
  --precomputed_images moses/train_images.npy

python train.py -pb --epochs 25 -i moses/test.smi --precomputed_values moses/test_sa.npy --lr 5e-5 -b 512 -w 2 -p sa \
  -t 1 --nheads 1 --dropout_rate 0.1 -o saved_models/moses_sa.pt --amp O2 --rotate --width 128 --depth 2  \
  --precomputed_images moses/test_images.npy --eval_train

python train.py -pb --epochs 25 -i moses/test_scaffold.smi --precomputed_values moses/test_scaffold_sa.npy --lr 5e-5 -b 512 -w 2 -p sa \
  -t 1 --nheads 1 --dropout_rate 0.1 -o saved_models/moses_sa.pt --amp O2 --rotate --width 128 --depth 2  \
  --precomputed_images moses/test_scaffold_images.npy --eval_train