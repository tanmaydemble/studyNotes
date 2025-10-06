Distributional semantics v/s denotational semantics

Distributional semantics focuses on gathering the meaning of a word by understanding other words around a given word whereas denotational semantics is focused with getting word meaning by associating a vector with a given word.

Word vectors

If two words have something to do with each other then they will look similar and their dot product will be large. This is how we know if a word embedding algorithm is good, that is, if there are similar words then their dot product is higher.

Word 2 vec algorithm

Each word in a fixed vocabulary is represented by a vector

Go through each position t in the text which has a center word c and a context o

Use the similarity of the word vectors for c and o to  calculate the probability of o given c

Keep adjusting the word vectors to maximize this probability

In practice, instead of maximizing this probability, we minimize the negative log likelihood of this probability as its easier to deal with summation than products

Soft max function

Max because it amplifies the probability of the largest x_i

Soft because it still assigns some probability to smaller x_i

This function turns a bunch of real numbers into a probability distribution - takes in real numbers and returns a number between 0 and 1

Parameters for the model

Theta represents all the model parameters in one long vector

Each word has two vectors - one as a center word and one as a context word

If we have V words in our vocabulary and each word is represented in d dimensions then theta will be a vector in R^2*d*V space.

Larger vs shorter context windows

Larger context windows capture contextual meaning whereas shorter context windows capture syntactic properties. For example, a syntactic property is that is or the is followed by a noun.

The acutal - expected interpretation of the word2vec algorithm

- The model predicts which words are likely to appear near a given context word.
- It does this by assigning a probability to every word in the vocabulary (all possible words).
- For each word, the model has a vector (think: a set of numbers that help describe the word’s meaning mathematically).
- The model calculates a weighted sum: for every word, multiply the probability it assigned to that word by that word’s vector, and then sum it all up.
- Updating: The vector for the context word (vcvc) is then adjusted to be more like the actual word that was observed (the real word that appeared next to it in the training data), and less like what the model expected (the weighted combination of all possible words).
- In short:  
    The model learns by shifting the context word’s vector closer to the true/real word’s vector, and away from its “average guess” for all words.

Analogy:

Imagine you’re trying to guess your friend’s favorite fruit. You make a list of all fruits and how likely you think each is (probability). When you find out the answer is “banana,” you update your expectations to move closer toward “banana” and away from your previous guesses

Absolutely! Here’s a self-contained summary you can put directly into your notes:

Word2Vec Skip-Gram: Softmax vs. Negative Sampling — Key Points

1. Softmax in Skip-Gram

- The classic skip-gram model predicts the probability of a context word www given a center word ccc using the softmax function:  
    p(w∣c)=exp⁡(uwTvc)∑w′∈Vexp⁡(uw′Tvc)p(w \mid c) = \frac{\exp(u_w^T v_c)}{\sum_{w' \in V} \exp(u_{w'}^T v_c)}p(w∣c)=∑w′∈Vexp(uw′Tvc)exp(uwTvc)
- Training goal: Maximize the log probability for real (center, context) pairs.
- Pulls the correct word pair together: Increases uwTvcu_w^T v_cuwTvc for observed context pairs (aligns their vectors).
- Pushes all other words away: Tries to decrease similarity uw′Tvcu_{w'}^T v_cuw′Tvc for all w′≠ww' \neq ww′=w by keeping the denominator small (dis-aligns non-co-occurring words).
- Partition function: The denominator, normalizing over the entire vocabulary, is called the partition function. It is computationally expensive because it sums over every word.

2. Negative Sampling

- Objective: Instead of using softmax, trains the model by:

- Maximizing similarity (via sigmoid, σ\sigmaσ), for the real (center, context) pair.
- Minimizing similarity for a small set of “negative” (random, incorrect) word pairs.

- Equation: Maximizes  
    log⁡σ(uoTvc)+∑l=1klog⁡σ(−ulTvc)\log \sigma(u_o^T v_c) + \sum_{l=1}^k \log \sigma(-u_l^T v_c)logσ(uoTvc)+l=1∑klogσ(−ulTvc)  
    where ooo is the real context word and the ulu_lul are negative samples.
- Sigmoid σ\sigmaσ: Squashes outputs to the range (0,1), acting like a “real vs. fake pair” probability.
- Use of log: Ensures stable optimization and prevents overestimating the probability of frequent words.
- Key benefit: Only compares the real word pair against a handful of randomly-sampled negatives on each update (not all words), dramatically speeding up training.

3. Comparison of Approaches

|   |   |   |
|---|---|---|
||Softmax (Old)|Negative Sampling (New)|
|Uses partition function?|Yes (over all of VVV)|No (just a few negatives)|
|Probability function|Softmax|Sigmoid|
|Trains with|Every word in vocab|Real + sampled negatives|
|Speed|Slow (with large vocab)|Much faster|
|Pulls/pushes pairs|Yes, for all words|Yes, but only a subset|

In essence:

- Softmax pulls real pairs together and pushes all other words away—but is slow due to the partition function.
- Negative sampling swaps softmax for sigmoid + log, “pulls” the right context pair together, “pushes” sampled negative pairs away, and trains much faster.