for c in 4
do
    python train.py --train_dataset ./RITE/train/ --val_dataset ./RITE/validation/ --epoch 800 --save_freq 20 --size $c
done
