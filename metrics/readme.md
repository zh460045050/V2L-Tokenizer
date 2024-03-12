# Perceptual metrics

## VGG distance
计算预测图像和groundtruth之间的特征距离，这里可以可选的距离有L1（mode=1）和MSE（mode=2）
```shell
cd vgg_distance
python vgg_distance.py -p=path_pred -t=path_groundtruth -m=2
```
Note:  
1) 这里读取图片的函数是*read_data*，这里是默认预测图像和groundtruth图像的命名是一致的，如果不是一致的，可以自行修改读取图像的方法，但仍然需要保证预测图像和groundtruth得一一对应。  
2) 这里需要预先下载VGG19模型，下载地址：链接：https://pan.baidu.com/s/1e3wn_bj5fwkZV1VVlSd5dg (提取码：pwhg)，下载后解压会出现一个vgg19文件夹，将此文件夹放在跟vgg_distance.py同一个文件夹下即可（同级目录）。  

## Inception Score (IS)
代码参考于：https://github.com/sbarratt/inception-score-pytorch  

```shell
cd inception_score
python inception_score.py -p=path_pred
```

## FID score
代码参考于：https://github.com/mseitzer/pytorch-fid  

```shell
cd fid_score
python fid_score.py path_pred path_groundtruth --batch-size=10 --dims=2048 -c=0
```
Note: *-c* 是指是否用显卡，如果不指定默认就是不使用，如果指定就需指定使用第几块卡，这里0是指使用第0块卡
