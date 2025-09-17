#### How to evaluate machine translation
##### BLEU
- Bilingual Evaluation Understudy
- Compares machine written translation to one or several human written translations and computes a similarity score based on:
	- Geometric mean of n-gram precision (usually for 1,2,3,4, grams)
	- There is also a brevity penalty that penalizes outputs that are smaller than reference texts
###### Disadvantages:
- There are many valid ways to translate a sentence
- A good translation can get a poor BLEU score because it odes not have overlap with the human translation
- You could translate the easy part of the sentence but leave out the hard part thus gaming the score.
##### BLEU Formula
The core BLEU formula is:

$$BLEU=BP × exp⁡(∑_{n=1}^Nw_nlog⁡(p_n))$$
- BP: Brevity Penalty, penalizes overly short translations
- p_n: Modified n-gram precision for n-grams of size n
- w_n: Weight for each n-gram order, typically 1NN1 for N-gram sizes (often N=4)
- N: Maximum n-gram size (commonly 4)[](https://www.geeksforgeeks.org/nlp/nlp-bleu-score-for-evaluating-neural-machine-translation-python/)
##### Modified N-gram Precision (pn)
pn=Clipped Count of matching n-grams / Total candidate n-grams
The candidate’s n-gram matches are **clipped** by their maximum count in any reference to avoid overcounting repeated words.[](https://www.geeksforgeeks.org/nlp/nlp-bleu-score-for-evaluating-neural-machine-translation-python/)
#### Brevity Penalty (BP)
BP= 1 if c > r, exp⁡(1−r/c) otherwise
 c is the length of the candidate output, r is the reference length (typically the closest reference length).

#### Attention
The theory of this was built in 2014 as opposed to other ML ideas which were proposed in the last decade.
For neural machine translation, the sentence in one language was converted into a single hidden layer which was fed to the other neural network that generates the sentence in another language. However, one hidden layer is too short to store so much information.

On each step of the decoder we are going to insert connections to the encoder to look up particular words in the sentence. 
We compare the hidden state we give to the decoder with the hidden state at each step of the encoder to calculate an attention score. Put them into a softmax and get a probability distribution of different words in the sentence. Use this weighting to get a weighted average of the different encoder states. Take that attention output and combine it with the hidden state of our decoder and use both of them together to generate an output vector which we put through our softmax to generate the first word of our translation. 

A simple way to calculate the attention is to take the dot product of the hidden state of the encoder and the hidden state of the decoder and put a softmax on that.
Use the probabilities from the softmax to get a weighted sum of the encoder hidden states. Finally, concatenate this output with the decoder hidden state and proceed as in the non attention seq2seq model.

- Attention is more human like process of translation as you can look back at the source sentence while translating rather than needing to remember it all
- Solves the bottleneck problem as the decoder can look directly at source and bypass the bottleneck. The bottleneck problem that attention solves in neural networks is the limitation imposed by relying on a fixed-length context vector to represent the entire input sequence, which particularly affects performance on long or complex inputs.
- Helps solve the vanishing gradient problem. Attention helps connect the decoder to different encoding hidden states. Hence, memory of past words is available when needed. 
- Provides interpretability. By looking at where the attention of the model is at, we can understand what it is translating. 

#### Multiple ways of calculating attention scores
Dot product doesn't always work great because the hidden state of an LSTM is its complete memory (26:07). This memory contains a lot of different information, some of which is relevant for the current translation step and some for future context or grammar.
When you use a dot product, it takes the entire hidden state into account. This means it tries to match all the information in the decoder's hidden state with all the information in the encoder's hidden state. However, you often only want to focus on the parts of the memory that are immediately useful for finding relevant information in the source sentence for the current word being translated

###### Multiplicative attention
Here, we stick a matrix between the current hidden state of the decoder and the hidden state of the encoder. This matrix helps learn the parameters. The matrix learns which parts of the generator hidden state should we be looking to find relevant information.

###### Reduced rank multiplicative attention
Instead of having a single big matrix, form it as a low rank matrix. A low rank matrix is one whose rank (the number of linearly independent rows or columns) is much smaller than its dimensions. In attention mechanisms, high-dimensional weight matrices are typically decomposed into two smaller matrices whose product approximates the original matrix.

