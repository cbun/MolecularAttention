# Molecular Image Attention Models



### Covid Data

If images have not been generated yet -- from raw data .csv file split the smiles column into a separate .csv file and then generate images with:
```
python generate_images.py -i <input smiles csv file> -o <output image npy file>
  ```
Then create data directories for each target with:
```
python process.py -r <raw csv file> -i <generated images .npy file>
```
Each directory will contain a <target_name>.smi file with smiles, a <target_name>.reg.npy file with dock scores, and a <target_name>.images.npy file with images

Training:
```
python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values <dir>/<target_name>.reg.npy \
                        --lr 8e-5 -w 3 -i <dir>/<target_name>.smi -o <dir>/model.pt --metric_plot_prefix <dir>/<target_name> \
                        --nheads 1 --dropout_rate 0.15 -- amp O2  --precomputed_images <dir>/<target_name>.images.npy  --width 128 \
                        --depth 2 -r 0 --scale
```        

Inference:
```
# inference on test data                                                                                                             
python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values <dir>/<target_name>.reg.npy \
           --lr 8e-5 -w 3 -i <dir>/<target_name>.smi -o <dir>/model.pt --nheads 1 --dropout_rate 0.15 --amp O2 \
           --precomputed_images <dir>/<target_name>.images.npy --width 128 --depth 2 -r 0 --scale --eval_test \
           --output_preds <dir>/out_test > <dir>/log_test.txt

# plot enrichment surface                                                                                                           python res.py <dir>/out_test.npy <dir> <target_name>
 ```
