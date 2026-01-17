Build the brain points to remember :



"fashion\_mnist = tf.keras.datasets.fashion\_mnist"



tf.keras.datasets → TensorFlow’s built-in datasets.



fashion\_mnist → A dataset of clothing images instead of digits.



You are assigning the dataset reference to a variable called fashion\_mnist.



Downloads the dataset automatically (only first time).



Loads it into memory.



Returns two pairs:



Training data



Validation/Test data



train\_images



Shape: (60000, 28, 28)



60,000 grayscale images



Each image is 28×28 pixels



👉 These are used to teach the ANN brain.

train\_labels



Shape: (60000,)



Each label is a number 0–9



Represents the clothing category



valid\_images



Shape: (10000, 28, 28)



Used to test how good the brain is



Model has never seen these during training



🔹 valid\_labels



Shape: (10000,)



Correct answers for validation images









```

| Label | Clothing Item |

| ----- | ------------- |

| 0     | T-shirt / Top |

| 1     | Trouser       |

| 2     | Pullover      |

| 3     | Dress         |

| 4     | Coat          |

| 5     | Sandal        |

| 6     | Shirt         |

| 7     | Sneaker       |

| 8     | Bag           |

| 9     | Ankle Boot    |



```

Input



Image pixels (28×28 → flattened to 784)



Learning



ANN learns shapes, edges, textures of clothes



Output



Predicts one of 10 clothing categories



tf



Short for TensorFlow



The main deep learning framework



🔹 keras



Keras was originally a separate deep learning library



TensorFlow officially integrated it → tf.keras



📌 Today, tf.keras is the recommended way to build models in TensorFlow.



What does tf.keras provide?

1️⃣ Layers (brain building blocks)

tf.keras.layers.Dense

tf.keras.layers.Conv2D

tf.keras.layers.Dropout





These are like neurons and connections.





Models (brain structure)

tf.keras.Sequential

tf.keras.Model





Used to assemble layers into a neural network.



3️⃣ Datasets (ready-made data)

tf.keras.datasets.mnist

tf.keras.datasets.fashion\_mnist





Used for training ANN/CNN models.



Optimizers (how the brain learns)

tf.keras.optimizers.Adam

tf.keras.optimizers.SGD





They update weights during training.







Loss Functions (measure mistakes)

tf.keras.losses.SparseCategoricalCrossentropy





Tells the model how wrong it is.



6️⃣ Metrics (performance tracking)

tf.keras.metrics.Accuracy





Used to measure accuracy, precision, recall, etc

```

| Without tf.keras | With tf.keras    |

| ---------------- | ---------------- |

| Complex math     | Simple API       |

| Manual backprop  | Automatic        |

| Slow development | Fast             |

| Hard to debug    | Clean \& readable |

```

Computers understand only numbers (0s and 1s).



Images are converted into numbers.



A neuron is just a small math function that:



takes numbers in



multiplies them by weights



adds them up



gives an output



This is how we build an artificial brain.



Math version (very simple)

y = mx + b





x → input (information coming in)



m → weight (how important the input is)



b → bias (extra adjustment)



y → output (decision signal)



📌 Learning = adjusting m and b so output becomes more accurate.



Step 1: From Human Neuron → Math Neuron

Biological idea



Dendrites → receive signals



Cell body → processes



Axon → sends output



Math version (very simple)

y = mx + b





x → input (information coming in)



m → weight (how important the input is)



b → bias (extra adjustment)



y → output (decision signal)



📌 Learning = adjusting m and b so output becomes more accurate.







Real data is not one value — it’s MANY values



Images don’t have one x, they have many pixels.



So the equation becomes:



y = w0·x0 + w1·x1 + w2·x2 + ... + b



What this means:



x0, x1, x2... → pixel values



w0, w1, w2... → importance of each pixel



b → bias



Each pixel gets its own weight.





Step 3: Why 784 weights?



Each image is 28 × 28 pixels



Total pixels =



28 × 28 = 784





So:



784 pixel values → 784 inputs



Each input → 1 weight



👉 One neuron has 784 weights + 1 bias



Pixel values:



0 → black



255 → white







One neuron is NOT enough



A single neuron outputs just a number.



But we want:



“Is this a shirt, trouser, shoe, bag…?”



So we do this 👇



✨ Idea: One neuron per class



Fashion-MNIST has 10 clothing categories.



So:



10 neurons



Each neuron “votes” for one class



The neuron with the highest output wins





Keras makes this easy



Instead of writing all this math manually, we use Keras.



Keras is:



A friendly tool that builds neural networks for us.



🧱 Step 6: Understanding the two layers (VERY IMPORTANT)

🔹 1. Flatten Layer 

images are like this:



\[

&nbsp;\[pixel, pixel, pixel],

&nbsp;\[pixel, pixel, pixel],

&nbsp;\[pixel, pixel, pixel]

]





Neurons want:



\[pixel, pixel, pixel, pixel, ...]





📌 Flatten converts 28×28 → 784 values





2\. Dense(10) Layer



“Dense” means fully connected



Each neuron:



sees all 784 pixels



has 784 weights



10 means:



10 neurons



10 outputs (one per clothing type)





7: The actual model (simple view)

model = tf.keras.Sequential(\[

&nbsp;   tf.keras.layers.Flatten(input\_shape=(28, 28)),

&nbsp;   tf.keras.layers.Dense(10)

])



What happens when an image enters?



Image → Flatten → 784 numbers



Numbers → Dense layer



Each neuron computes:



y = w·x + b





Highest output → predicted classFinal Mental Picture (Remember This)

Image (28×28)

&nbsp;     ↓

Flatten (784 numbers)

&nbsp;     ↓

10 Neurons (each votes)

&nbsp;     ↓

Highest vote = answer



“Each image is converted into 784 pixel values, flattened into a single vector, and passed into a dense layer where each neuron learns a weighted sum of pixels to classify the image into one of ten categories.”



Sparse Categorical Cross-Entropy is one of the most important loss functions in deep learning.





Categorical



Used when you have more than 2 classes



Example:



Digits 0–9



Fashion classes (10 clothing types)





Sparse



Labels are given as integers, not one-hot vectors



Example:



Label = 3





instead of



\[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]





✔ Much simpler

✔ Less memory



Cross-Entropy



Measures the difference between:



True label



Predicted probability



👉 It heavily penalizes confident wrong predictions.





True label:

Shirt → label = 6



Model prediction:

\[0.01, 0.02, 0.01, 0.05, 0.10, 0.01, 0.70, 0.05, 0.03, 0.02]





Model gives 70% probability to class 6 ✔



Loss:

Loss = -log(0.70) ≈ 0.36





Lower loss = good prediction ✅



❌ Wrong prediction example:

Probability for class 6 = 0.01

Loss = -log(0.01) ≈ 4.6





Huge penalty 🚨



```

| Situation             | Use |

| --------------------- | --- |

| Multi-class problem   | ✔   |

| Labels are integers   | ✔   |

| Softmax output        | ✔   |

| Fashion-MNIST / MNIST | ✔   |





| Loss Function                   | Label Format    |

| ------------------------------- | --------------- |

| `CategoricalCrossentropy`       | One-hot encoded |

| `SparseCategoricalCrossentropy` | Integer labels  |



\# Sparse

y = 5



\# Categorical

y = \[0,0,0,0,0,1,0,0,0,0]







```



How we use it in TensorFlow

model.compile(

&nbsp;   optimizer='adam',

&nbsp;   loss=tf.keras.losses.SparseCategoricalCrossentropy(from\_logits=True),

&nbsp;   metrics=\['accuracy']

)



from\_logits=True



Means:



Output layer does NOT use softmax



TensorFlow applies softmax internally (numerically stable)





If your model ends with:



Dense(10, activation='softmax')





Then:



loss = tf.keras.losses.SparseCategoricalCrossentropy(from\_logits=False)





“Sparse Categorical Cross-Entropy is a loss function used for multi-class classification where labels are integer encoded, and it measures how far the predicted probability distribution is from the true class.”



Mental Picture (Remember This)

True class → pick its probability → apply -log → loss









n this case, we use a Sparse Categorical Cross-Entropy loss function, which is specifically designed for multi-class classification problems. The term sparse indicates that the true labels are provided as integer indices rather than one-hot encoded vectors. The word categorical shows that the function is intended for classification tasks involving multiple categories. Cross-entropy measures how well the model’s predicted probability distribution matches the true label, and it penalizes the model more severely when it makes confident but incorrect predictions. In extreme cases, if the model is completely confident and still wrong, the loss value approaches negative infinity.



The from\_logits parameter is used when the model’s output layer produces raw numerical values instead of probabilities. In this case, the loss function internally applies a softmax operation to convert these values into probabilities, which represent the model’s confidence for each category. This loss function is well suited for our problem because it evaluates the outputs of all neurons simultaneously, ensuring that only the neuron corresponding to the correct class is encouraged while suppressing incorrect ones. Since multiple neurons cannot all represent the correct label at the same time, this approach helps the model learn clear and accurate class distinctions.



In addition to the loss function, we also monitor performance metrics such as accuracy to better understand how well the model is learning. A low loss value does not always guarantee high accuracy, so tracking multiple metrics provides a more complete picture of the model’s training progress and overall performance.





How did the model do? B-? To give it credit, it only had 10 neurons to work with. Us humans have billions!



The accuracy should be around 80%, although there is some random variation based on how the flashcards are shuffled and the random value of the weights that were initiated.



Prediction

Time to graduate our model and let it enter the real world. We can use the predict method to see the output of our model on a set of images, regardless of if they were in the original datasets or not.



Please note, Keras expects a batch, or multiple datapoints, when making a prediction. To make a prediction on a single point of data, it should be converted to a batch of one datapoint.



Below are the predictions for the first ten items in our training dataset.





































