Images in part 2 don't make sense.
In backprop for convolution, why don't we divide dw by N
Check backprop for convolution by changing stride to 2 - DONE
What filters to plot for visualization in convnet?

Checklist:

Target validation accuracies for various models:

Softmax - 25 
- Numpy 27.5 - Loss curve goes down but images don't make sense
Two layer NN - 30
 - (I altered the actual one a bit here - need to revert it back) 48%
Convnet - 50
- 52%, didn't plot yet but bad images


Bonus question - 81
--kernel-size 3 --hidden-dim 5 --epochs 10 --weight-decay 0.0 --momentum 0.0 --batch-size 512 --lr 0.001 - 42%
--kernel-size 3 --hidden-dim 10 --epochs 10 --weight-decay 0.0 --momentum 0.0 --batch-size 512 --lr 0.001 - 49%
--kernel-size 3 --hidden-dim 10 --epochs 10 --weight-decay 0.95 --momentum 0.9 --batch-size 512 --lr 0.001 - 50%
--kernel-size 3 --hidden-dim 20 --epochs 5 --weight-decay 0.95 --momentum 0.9 --batch-size 512 --lr 0.001 - 50%
--kernel-size 5 --hidden-dim 20 --epochs 5 --weight-decay 0.95 --momentum 0.9 --batch-size 512 --lr 0.001 - 52%



