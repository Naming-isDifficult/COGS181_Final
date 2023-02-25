# Enhanced-AdaIN-with-U-net
Final project for COGS 181 WI23. This project proposed a possible way to improve AdaIN proposed by Xun and Serge[^1] Belongie by replacing the encoder-decoder architecture with U-net proposed by Olaf et. al.[^2]. Although U-net is designed for image segmentation mission, its auto-encoder-like architecture makes it possible for the U-net to perform auto-encoding.<br>

## Reference
[^1]: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)<br>
[^2]: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Pending Issues

## TO-DO
 - [x] Upsampling layer
   - [x] Test (input-output shape)
 - [x] Downsampling layer
   - [x] Test (input-output shape)
 - [x] AdaIN layer
   - [x] Test (input-output shape)
   - [x] AdaIN
 - [x] U-net Model
   - [ ] Trainer/Appilcation
 - [x] UAdaIN Model
   - [ ] Trainer/Application
 - [ ] Data pipeline
   - [ ] DataSet
   - [ ] DataLoader
 - [ ] Pretraining
 - [ ] Training