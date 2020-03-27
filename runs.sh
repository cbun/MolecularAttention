#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0327_freeze.pt -i t1/ml.adrp_p1_smiles.csv --precomputed_images t1/ml.adrp_p1_images.npy  --precomputed_values t1/ml.adrp_p1.dock.npy --metric_plot_prefix adrp_0327_freeze --mae --width 128 --depth 1 --forward --freeze > output_0327_freeze.txt



#CUDA_VISIBLE_DEVICES=0 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0327_dep2.pt -i t1/ml.adrp_p1_smiles.csv --precomputed_images t1/ml.adrp_p1_images.npy  --precomputed_values t1/ml.adrp_p1.dock.npy --metric_plot_prefix adrp_0327_dep2 --mae --width 128 --depth 2 > output_0327_dep2.txt


CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0327_fforward.pt -i t1/ml.adrp_p1_smiles.csv --precomputed_images t1/ml.adrp_p1_images.npy  --precomputed_values t1/ml.adrp_p1.dock.npy --metric_plot_prefix adrp_0327_fforward --mae --width 128 --depth 1 --forward > output_0327_fforward.txt

#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp.pt -i t1/smiles_valid --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_p1_valid.npy --metric_plot_prefix adrp_pocket_1 --mae --width 128 --depth 2  

#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 50  -w 0 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0326_nuc.pt -i t1/smiles_trim.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_p1_trim.npy --metric_plot_prefix adrp_p1_0326_nuc --mae --width 128 --depth 4

#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 350 --epochs 100  -w 0 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0327_100e.pt -i t1/smiles_trim.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_p1_trim.npy --metric_plot_prefix adrp_p1_0327_100e --mae --width 128 --depth 1 --forward

#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 50  -w 0 -r 0  --nheads 0 --dropout_rate 0.1  --lr 8e-5 -o saved_models/model_adrp_0327_dr01.pt -i t1/smiles_trim.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_p1_trim.npy --metric_plot_prefix adrp_p1_0327_dr01 --mae --width 128 --depth 1 --forward    


#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 40  -w 0 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0327_dep5.pt -i t1/smiles_trim.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_p1_trim.npy --metric_plot_prefix adrp_p1_0327_dep5 --mae --width 64 --depth 5

#CUDA_VISIBLE_DEVICES=1 python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 30  -w 0 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_0326_30e.pt -i t1/smiles_trim.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_p1_trim.npy --metric_plot_prefix adrp_p1_0326_30e --mae --width 128 --depth 1

#python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 15  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-4 -o saved_models/model_adrp_02.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_02 --mae --width 128 --depth 4 --no_pretrain


#python train_dock_images.py -pb --rotate -t 1 -b 350 --epochs 30  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 5e-5 -o saved_models/model_adrp_03.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_03 --mae --width 128 --depth 4 --no_pretrain

#python train_dock_images.py -pb --rotate -t 1 -b 350 --epochs 30  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-4 -o saved_models/model_adrp_04.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_04 --mae --width 128 --depth 4 



#python train_dock_images.py -pb --rotate -t 1 -b 128 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.15  --lr 8e-5 -o saved_models/model_adrp_05.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_05 --mae --width 128 --depth 2 

#python train_dock_images.py -pb --rotate -t 1 -b 256 --epochs 30  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-4 -o saved_models/model_adrp_06.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_06 --mae --width 128 --depth 1


#python train_dock_images.py -pb  --rotate -t 1 -b 128 --epochs 10  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-4 -o saved_m\
#odels/model_adrp_07.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy \
# --metric_plot_prefix adrp_07 --mae --width 64 --depth 3 

#python train_dock_images.py -pb  --rotate -t 1 -b 512 --epochs 10  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-4 -o saved_models/model_adrp_08.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_08 --mae --width 64 --depth 3 -g 2 

#python train_dock_images.py -pb  --rotate -t 1 -b 350 --epochs 100  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-3 -o saved_models/model_adrp_09.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_09 --mae --width 64 --depth 4 

#CUDA_VISIBLE_DEVICES=0 python train_dock_images.py -pb  --rotate -t 1 -b 256 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.1  --lr 1e-5 -o saved_models/model_ad\
#rp_10.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_10 --mae --width 64 --depth 3

#python train_dock_images.py -pb  --rotate -t 1 -b 350 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.15  --lr 1e-5 -o saved_models/model_ad\
#rp_11.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix \
#adrp_11 --mae --width 32 --depth 5

#CUDA_VISIBLE_DEVICES=2 python train_dock_images.py -pb  --rotate -t 1 -b 350 --epochs 50  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 6e-5\
# -o saved_models/model_ad\
#rp_12.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_pre\
# fix adrp_12 --mae --width 128 --depth 4

#python train_dock_images.py -pb  --rotate -t 1 -b 512 --epochs 25  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 6e-5\
#		    -o saved_models/model_adrp_13.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_13 --mae --width 128 --depth 4 --no_pretrain

#python train_dock_images.py -pb --rotate -t 1 -b 350 --epochs 100  -w 1 -r 0  --nheads 0 --dropout_rate 0.2  --lr 1e-3 -o saved_models/model_adrp_14.pt -i t1/smiles_dock_v2.csv --precomputed_images t1/image_dock_v2.npy  --precomputed_values t1/adrp_pocket_1_dock.npy --metric_plot_prefix adrp_14 --mae --width 128 --depth 2

