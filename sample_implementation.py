from AnomalyDetector import *


if __name__ == '__main__':
    ds_path = '{dataset local path}'  # https://paperswithcode.com/dataset/mvtecad
    item = 'carpet'
    split = 'train'
    data_path = os.path.join(ds_path, item, split)

    # Dataset utils
    resize = (224, 224)
    blur_size = None
    rand_resize_scale = (0.8, 1)
    translate = (0.1, 0.1)

    # Training variables
    batch = 16
    learning_rate = 0.001
    epochs = 100
    patience = 5
    split_per = [0.85, 0.15]

    inst = AnomalyDetector(data_path=data_path, resize=resize, epochs=epochs, translate=translate,
                           rand_resize_scale=rand_resize_scale, blur_size=blur_size, split_per=split_per,
                           lr=learning_rate)

    # TRAINING --------------------------------------------------------------------------------------------------------
    is_train = True
    if is_train:
        inst.train()

    # TESTING --------------------------------------------------------------------------------------------------------
    split = 'test'
    data_path = os.path.join(ds_path, item, split)
    test_ds_utils = DatasetUtils(data_path=data_path, resize=resize)
    test_ds = test_ds_utils.create_dataset()
    model_path = '{your saved model path}'
    inst.test(model_path=model_path, data=test_ds)
