python train.py -pb --rotate --mae -t 16 -b 128 --epochs 50 --precomputed_values qm8/qm8_values.npy \
  --lr 5e-5 -w 1 -i qm8/qm8.smi -o saved_models/qm8_4.pt --nheads 0 --dropout_rate 0.1 --amp O1 \
  --precomputed_images qm8/qm8_images.npy --width 256  --depth 4 --cv 4 -r 0 -p custom
