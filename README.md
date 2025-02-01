# Image Anomaly Detection (MVTecAD Dataset)

## 1.0 About

This project aims to build an autoencoder model that can detect anomalies on images, typically found on industrial inspections.

<br/>

![image](https://github.com/user-attachments/assets/8fbd15c2-32e2-4364-8c4f-51be112fcbd2)

<br/>

## 2.0 How it works?

![image](https://github.com/user-attachments/assets/576e951e-d2c0-46a5-89b3-91bf17d0d7cd)

1. Autoencoder model is strictly trained on "Good" *(No anomalies)* data so that it will always try to reconstruct the image without anomalies.
2. The reconstructed image will then be compared to the input image and measure the Mean Square Error (MSE).
3. MSE score will be used to determine if the input image have anomalies.
4. Output will be **0** for "Good" and **1** for "Anomaly".

## 3.0 sample implementation
You can check out this [sample_implementation.py](https://github.com/lloydaxeph/image_anomaly_detection/blob/main/sample_implementation.py) file.
```python
# TRAINING ---
inst = AnomalyDetector(data_path=data_path, resize=resize, epochs=epochs, translate=translate,
                           rand_resize_scale=rand_resize_scale, blur_size=blur_size, split_per=split_per,
                           lr=learning_rate)
inst.train()
# TESTING ---
inst.test(model_path=model_path, data=test_ds)
```
