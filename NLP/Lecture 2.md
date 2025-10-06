stochastic gradient descent is faster than regular gradient descent as we are calculating the gradient over a smaller sample from the data as compared to the whole data set. Also, while the stochastic method is an approximation, it adds random noise and jiggle to the data which helps the neural nets generalize better.

initialize your vectors with random small numbers because if everything starts out the same at 0 then you get these false symmetries which means you can't learn.

for our word 2 vec algorithm, the only parameters are the U and V matrices. The U matrix has the outside word vectors and the V matrix has the center word vectors. We take the dot product of the outside matrix with the center word and we apply SoftMax over that to get a probability distribution which represents which outside words are more likely. We then compare this with the actual outside word to get our error. This is a bag of words model. We don't know about the structure of sentence.

skipgram predicts context or outside words given center word

continuous bag of words predicts center word from the bag of context words

loss functions to be used for training

 1. SoftMax - sum over every word in the vocabulary

2. negative sampling - rather than evaluating the dot product for each word in the vocabulary, we train a logistic regression model that tells us that we like some words in the context of a center word and don't like certain other words in the context of the center word. we use the negative log sigmoid of the dot product instead of the SoftMax. In this loss function we take the positive of the dot product of outside word with the correct inside word and the negative of the rest of the words. we minimize this. for our random sample of words, to train the model on, we don't use a uniform distribution, rather we account for the frequencies of the words in the corpus. this is called the unigram distribution. do improve this, we raise the unigram probability by 3/4 power. raising by this power makes sure that we are somewhere in between of the uniform distribution and the most frequently used words distribution.

why not capture co-occurrence counts directly?

given a context window, we can create a co-occurrence count matrix. the size of this matrix is worse than the dimensions of a word vector. to reduce dimensionality, we use svd. here, any matrix can be written as a product of three matrices. the U and V matrices are orthogonal and made of unit vectors.

linear semantic component - technical term for the example with king minus man plus woman is queen in vectors.

intrinsic vs extrinsic evaluation

intrinsic

- fast to compute

- distant from downstream task

extrinsic

- real task like machine translation, document evaluation, question answering

- unclear if the subsystem is the problem or its interaction or other subsystems

word similarities

intrinsic word vector evaluation

asking people about how similar words are and then comparing this with what the model says about similarity and then calculating correlation

extrinsic word vector evaluation

named entity recognition

chris manning lives in palo alto

good word vectors should be able to identify that chris manning refers to a person and palo alto refers to a place

word senses

cluster word windows around words, retrain with each word assigned to multiple different clusters

learn word vectors for token clusters

or

use one word vector for different senses which is a weighted average or superposition of different senses. word meanings have a lot of nuance and cutting them into senses is artificial, even different dictionaries disagree on different senses of words.

standard math tells you that a single word vector cannot cover all meanings in a single vectors but sparse coding theory tells that we can separate out the senses from one word vector

named entity recognizer, label words as person, location or date - used in Wikipedia, you link a webpage to a named entity.

cross entropy loss - if we have a true probability distribution p and we are computing a probability distribution q then the cross entropy loss is the log of the model probability and the expectation of that under the true probability distribution

assuming a ground truth probability distribution that is 1 at the right class and 0 everywhere else, then the only term left as our loss function is the negative log probability of the true class y_i

neural networks is multiple logistic regressions in parallel and one after the other