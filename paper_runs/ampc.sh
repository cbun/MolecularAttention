python train.py -pb --rotate --mae -p custom -t 1 -b 512 --epochs 25 --amp O2 -w 1 -r 0 --depth 2 --width 128 --nheads 1 --dropout_rate 0.1 \
 --lr 1e-4 -o saved_models/ampc.pt -i ampc/ampc_smiles.smi --precomputed_images ampc/ampc_images.npy \
 --precomputed_values moses/ampc_values.npy