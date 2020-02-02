python train.py -pb --rotate --mae -t 16 -b 128 --epochs 25 --precomputed_values qm8/qm8_values.npy \
  --lr 1e-4 -w 1 -i qm8/qm8.smi -o saved_models/qm8.pt --nheads 64 --dropout_rate 0.15 --amp O1 \
  --precomputed_images qm8/qm8_images.npy --width 128  --depth 5 --cv 1 -r 0 --no_pretrain