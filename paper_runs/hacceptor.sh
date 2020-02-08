python train.py --rotate -pb --amp O2 -w 1 --nheads 0 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
 --epochs 15 -b 512 --lr 1e-4 -i moses/train.smi --precomputed_values moses/train_hacceptor.npy \
  --precomputed_images moses/train_images.npy -o saved_models/moses_hacceptor_noattn.pt

python train.py --rotate -pb --amp O2 -w 1 --nheads 0 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
 --epochs 15 -b 512 --lr 1e-4 -i moses/test.smi --precomputed_values moses/test_hacceptor.npy \
  --precomputed_images moses/test_images.npy -o saved_models/moses_hacceptor_noattn.pt --eval_train  > moses_hacceptor_performance.txt

python train.py --rotate -pb --amp O2 -w 1 --nheads 0 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
  --epochs 15 -b 512 --lr 1e-4 -i moses/test_scaffold.smi --precomputed_values moses/test_scaffold_hacceptor.npy \
  --precomputed_images moses/test_scaffold_images.npy -o saved_models/moses_hacceptor_noattn.pt --eval_train  >> moses_hacceptor_performance.txt

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
# --epochs 15 -b 512 --lr 1e-4 -i moses/train.smi --precomputed_values moses/train_hacceptor.npy \
#  --precomputed_images moses/train_images.npy -o saved_models/moses_hacceptor.pt

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
# --epochs 15 -b 512 --lr 1e-4 -i moses/test.smi --precomputed_values moses/test_hacceptor.npy \
#  --precomputed_images moses/test_images.npy -o saved_models/moses_hacceptor.pt --eval_train  > moses_hacceptor_performance.txt

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
#  --epochs 15 -b 512 --lr 1e-4 -i moses/test_scaffold.smi --precomputed_values moses/test_scaffold_hacceptor.npy \
#  --precomputed_images moses/test_scaffold_images.npy -o saved_models/moses_hacceptor.pt --eval_train  >> moses_hacceptor_performance.txt


#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
# --epochs 15 -b 512 --lr 1e-4 -i moses/train.smi --precomputed_values moses/train_hacceptor.npy \
#  --precomputed_images moses/train_images.npy -o saved_models/moses_hacceptor_bw.pt --bw

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
# --epochs 15 -b 512 --lr 1e-4 -i moses/test.smi --precomputed_values moses/test_hacceptor.npy \
#  --precomputed_images moses/test_images.npy -o saved_models/moses_hacceptor_bw.pt --eval_train  --bw >> moses_hacceptor_performance.txt

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
#  --epochs 15 -b 512 --lr 1e-4 -i moses/test_scaffold.smi --precomputed_values moses/test_scaffold_hacceptor.npy \
#  --precomputed_images moses/test_scaffold_images.npy -o saved_models/moses_hacceptor_bw.pt --eval_train  --bw >> moses_hacceptor_performance.txt


#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
# --epochs 15 -b 512 --lr 1e-4 -i moses/train.smi --precomputed_values moses/train_hacceptor.npy \
#  --precomputed_images moses/train_images.npy -o saved_models/moses_hacceptor_nopt.pt --no_pretrain

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
# --epochs 15 -b 512 --lr 1e-4 -i moses/test.smi --precomputed_values moses/test_hacceptor.npy \
#  --precomputed_images moses/test_images.npy -o saved_models/moses_hacceptor_nopt.pt --eval_train --no_pretrain  >> moses_hacceptor_performance.txt

#python train.py --rotate -pb --amp O2 -w 1 --nheads 1 -r 0 --dropout_rate 0.1 -p hacceptor --depth 2 --width 128 -r 0 \
#  --epochs 15 -b 512 --lr 1e-4 -i moses/test_scaffold.smi --precomputed_values moses/test_scaffold_hacceptor.npy \
#  --precomputed_images moses/test_scaffold_images.npy -o saved_models/moses_hacceptor_nopt.pt --eval_train  --no_pretrain >> moses_hacceptor_performance.txt
