python train.py -pb --rotate --mae -p all -b 128 --epochs 25 --precomputed_values moses/train_descriptors.npy \
  --lr 1e-4 -w 1 -i moses/train.smi -o saved_models/moses_descriptors.pt --nheads 64 --dropout_rate 0.15 --amp O1 \
  --precomputed_images moses/train_images.npy --width 512  --depth 5 -r 0
