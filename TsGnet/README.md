# TsGNet

## Prepare datasets

## Training PearNet
The `config.json` file is used to update the training parameters.
To perform the standard K-fold crossvalidation, specify the number of folds in `config.json` and run the following:
```
chmod +x batch_train.sh
./batch_train.sh 0 /path/files
```
where the first argument represents the GPU id.

## Results
The log file of each fold is found in the fold directory inside the save_dir.   

