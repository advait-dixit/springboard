import numpy as np
import tensorflow as tf
import skimage


def load_data(label_directories):
    labels = []
    images = []
    for l in label_directories:
        for i in range(1, 300):
            images.append(skimage.data.imread("%s/%s%d.jpg" % (l, l, i)))
            labels.append(ord(l[0]) - ord('A'))
    return images, labels


images, labels = load_data(['A', 'B', 'C'])
images28 = [skimage.transform.resize(image, (28, 28)) for image in images]
# Convert 'images28' to an array
images28 = np.array(images28)

# Convert 'images28' to grayscale
images28 = skimage.color.rgb2gray(images28)

# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 3, tf.nn.relu)

# Define a loss function
# softmax is type of output layer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                    logits = logits))
# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss_value)

    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: images28})[0]

    # Calculate correct matches
    match_count = sum([int(y == y_) for y, y_ in zip(labels, predicted)])

    # Calculate the accuracy
    accuracy = float(match_count) / len(labels)

    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))
    print predicted
