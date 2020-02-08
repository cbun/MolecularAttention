python train.py -pb --rotate --mae -p custom -t 1029 --mae  -b 384 --epochs 25 --precomputed_values moses/train_descriptors.npy \
  --lr 8e-5 -w 2 -i moses/train.smi -o saved_models/moses_descriptors.pt --nheads 0 --dropout_rate 0.1 --amp O1 \
  --precomputed_images moses/train_images.npy --width 2096  --depth 3 -r 0 --imputer_pickle moses/moses_desc_imputer.pkl --no_pretrain --output_preds moses/desc_outut.npy --eval_test
