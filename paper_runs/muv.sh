python train.py -pb --rotate -p custom --mae -t 17 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 0 --dropout_rate 0.1 \
 --lr 1e-4 -o saved_models/muv.pt -i MUV/muv.smi --precomputed_images MUV/muv_images.npy \
 --precomputed_values MUV/muv_values.npy --mask MUV/muv_mask.npy
