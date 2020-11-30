'''
Ideas for better loss functions:

1. Soft Dice Loss
The Dice Coefficient is calculated for each of the different depth layers in the output

2. Weighted Pixel Wise CCE: https://arxiv.org/abs/1411.4038

'''
import numpy as np
import tensorflow as tf

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 

    axes = tuple(range(1, len(y_pred.shape))) 
    print(axes)
    numerator = 2. * np.sum(y_pred * y_true, axes)

    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    print(numerator.shape, denominator.shape)
    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon))

mids = [2, 5, 7, 9, 11, 13, 15]

def topLmismatch(y_true, y_pred):
	y_pred_reg = np.zeros((128, 128))
	for i in range(128):
		for j in range(128):
			y_pred_reg[i][j] = np.dot(y_pred[i][j], mids)
	print(y_pred_reg.argsort()[-3:][::-1])

def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in 
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


y_true = np.zeros((1, 128, 128, 7))
y_pred = np.zeros((1, 128, 128, 7))
y_true[0][0][0][0] = 1
y_pred[0][0][0][1] = 1
print(y_pred[0][0][0].shape)
# topLmismatch(y_true, y_pred)
print(gen_dice(y_true, y_pred))