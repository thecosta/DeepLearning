This model classifies CIFAR10 with no CNN.
All code done in tensorflow.

The required performance was to achieve a 40% accuracy on the tested model.

To train the model please do:
```
python3 classify.py train
```
To use the model to predict please do:
```
python classify.py test --image_path cat.png
```

Prediction of new image classes is not always correct since it is hard to do well with this type of model.
