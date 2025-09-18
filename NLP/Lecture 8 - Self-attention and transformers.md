#### Issues with recurrent models
###### Linear interaction distance
RNNs process information sequentially, either left-to-right or right-to-left. This means that words interact based on their linear proximity in the sentence. For example, "tasty pizza" has words that are close and can easily interact.
 For words that are far apart in a sentence but are still grammatically or semantically related (like "the chef" and "was" in a long sentence), the RNN has to perform many computational steps to connect them. This makes it difficult for the network to learn these long-distance dependencies because gradients have to propagate through many layers, leading to issues like vanishing gradients. This linear order isn't always the best way to understand sentence structure, as related words can be far apart. **Solution with attention**: Self-attention, which the lecture focuses on, helps solve this by allowing words to interact with any other word in the sequence, regardless of their distance, in a single operation.
 
###### Lack of parallelizability
RNNs process information sequentially, meaning the hidden state for a given time step (like word 5) cannot be computed until the hidden state for the previous time step (word 4) is known. This creates a chain of dependencies. You can't compute all parts of the sequence at once. The forward and backward passes in an RNN have a number of "unparallelizable operations" roughly equal to the sequence length. While GPUs are excellent at performing many operations simultaneously, they can't help much when there's a strict time dependency between operations. This means RNNs cannot fully leverage the parallel processing power of GPUs for long sequences.  Self-attention, in contrast, allows all words in a sequence to be processed simultaneously (in parallel), as the attention for each word can be computed independently without waiting for previous.

###### Self attention
Attention in LSTMs treats each word's representation as a query to access and incorporate information from a set of values. This was used in sentence translation, in decoders to access encoder information. The new way of doing things is applying **attention _within_ a single sentence** – this is called **self-attention**.

###### Query, Key and Value vectors
In the context of self-attention, a **query (Q)** is a vector representation of a word that is used to **ask for information** from other words in the sequence. - If you want to understand a particular word better, you form a "query" based on that word. This query vector is then compared against the **key** vectors of all other words in the sentence (including itself) to determine how relevant each of those other words is to the current word. - The more similar a key is to the query, the more "attention" will be paid to that word's corresponding **value** vector. So, for every word in a sentence, it generates its own unique Query vector, along with a Key and Value vector, to facilitate this internal "lookup" process

###### Attention is a fuzzy lookup
Attention is like a fuzzy lookup in a key value store. Attention is described as "fuzzy" in the video because, unlike a traditional lookup table or dictionary where a query must _exactly match_ a key to retrieve a specific value, attention operates in a **soft, probabilistic manner** in a vector space. 
**Soft Matching**: Instead of an exact match, attention computes a **similarity score** between your **query** and _all_ available **keys**. This similarity isn't just a yes/no, but a continuous value indicating how related they are.  These similarity scores are then converted into **weights** (often using a softmax function), which are typically between 0 and 1.  The final output is not a single value but a **weighted sum of all the values**.

![[Pasted image 20250917191553.png]]

w is a one hot vector of dimensions 1 * |V| so when we multiply it with E we get a vector of dimension d. 
We learn matrices Q and K separately because is becomes a low rank matrix of the matrix that would be the product of $q_i$ and $k_j$. This provides computation efficiencies. 

#### Fixing the no sequence order problem
There is no sequence order when looking at attention. Hence, we need to encode the order of the sentence in our keys, queries and values. Consider representing each sequence index as a vector.  Consider a vector $p_i$ which represents a sequence index in dimension d. To incorporate sentence order, just add the $p_i$ to the inputs. 
###### Sinusoidal position representations
Concatenate sinusoidal functions of varying periods in a vector. periodicity indicates that maybe absolute position isn't as important. 
Self-attention, by itself, processes words as a "set" and has no inherent understanding of their order. Sinusoidal functions create unique vectors for each position that can be added to the word embeddings, providing this crucial positional information.
**How they are formed**: For each dimension within the position vector (which has dimensionality D, like the word embeddings), a sine or cosine function is used. The period of these sine/cosine waves varies across the dimensions, allowing the combination of these waves to create a unique positional "fingerprint" for each word's index in the sequence.
**Intuition**: The core intuition is that **absolute position might not be as important as relative position**, and periodicity can capture this. 
- If you just give it an absolute number (like 1st word, 2nd word, 3rd word), the model learns that "1" means "first." But what if you encounter a sentence with 100 words, and your model was only trained on sentences up to 50 words? How does it know what "51st" means?
- **Sinusoidal functions** (sine and cosine waves) are periodic, meaning they repeat their pattern over time. By using different sine and cosine waves with **varying periods** across the different dimensions of the position vector, the idea is that:
    - The combination of these waves creates a unique "signature" for each position.
    - But more importantly, the _difference_ or _relationship_ between positions might be consistent even for unseen, longer sequences. For example, the "distance" between position 5 and position 10 might look similar to the "distance" between position 55 and position 60 in terms of the sinusoidal patterns, even if the absolute values are different.

This **periodicity** is supposed to give the model a sense of **relative position** and make it easier to **extrapolate** to sequences longer than those seen during training
**In practice**: While it's an early and still sometimes used method, the video notes that in practice, this particular intuition about extrapolation doesn't always work as expected. 
![[Pasted image 20250917204917.png]]
###### Position representation vectors learned from scratch
Let $p_i$ be all learnable parameters. Learn a matrix p which is d x n and let $p_i$ be a column of that matrix.  Add this to $x_i$. The advantage is that each position gets to be learned to fit the data but the con is that we can't extrapolate to indices outside 1,..., n as the matrix size is n.

#### Adding non linearities in self attention
self-attention alone, as initially presented, is primarily a **linear operation** because it mainly involves computing weighted averages of value vectors. If you were to stack multiple self-attention layers without anything else, you would essentially just be re-averaging value vectors, which doesn't add much **expressive power** or allow for the "deep learning magic" that makes neural networks so powerful. To address this, the solution is to add **non-linearities** through a **feed-forward network (FFN)** after each self-attention block.
![[Pasted image 20250917210644.png]]
Take the output of the attention calculation which is the $o_i$ vector and put it through a FFN and chain it together to understand deeper relationships. The FFN looks like:
![[Pasted image 20250917210850.png]]

#### Masking the future in self-attention
To use self-attention in decoders, we need to ensure that we can't peek at the future. At every timestamp we could change the set of keys and queries to include only the past words. However, this is slow and to enable parallelization, we mask out attention to future words by setting attention scores to -inf. So 
![[Pasted image 20250917211319.png]]
This ensures that if i which is the current word that we are looking at comes after j then we can use it as input and get scores otherwise it is -inf which becomes 0 when put through softmax. 
You need to mask attention for decoders because they are often used in **autoregressive tasks**, meaning they generate a sequence one element at a time, predicting the next word based on the previous ones.

#### Multi-head self attention
###### Sequence stacked form of attention
Same as the attention block but instead of treating each input vector separately, we create a matrix by concatenating the input vectors. This makes the computation on GPUs faster. 

The point of multi-head attention is to enable the model to attend to different aspects of the input sequence _independently_ and _simultaneously_. By splitting the `D` dimensions of the word vector into `H` parts (where `D/H` is the dimension of each head), each "head" can learn to focus on different types of relationships or information within the sequence.
For example, one head might learn to identify syntactic dependencies (like subject-verb agreement), while another might focus on semantic relationships or coreference (e.g., what "it" refers to). These "different things" are learned through separate sets of query, key, and value matrices for each head.
![[Pasted image 20250918173857.png]]
#### Scaled dot product
When dimensionality d becomes large, dot products become large as we are multiplying and adding more numbers. Because of this, inputs to the softmax function can be large making the gradients small. Hence, we divide the attention score by $sqrt(d/h)$.
1. **Softmax Saturation:** The softmax function takes these scores and turns them into probabilities (weights that sum to 1). If one score is significantly larger, the softmax will push its corresponding probability very close to 1, and all other probabilities very close to 0. This is called **saturation**.
2. **Flat Gradient:** When the output of a function (like softmax) is very close to 0 or 1, the gradient of that function becomes very small. Imagine a very steep hill (large gradient) that flattens out at the top or bottom (small gradient). Once the softmax output is "saturated," small changes in the input (the large dot product) barely change the output probability.

#### Residual connections
- **Easier Training:** They help models train much better, especially very deep ones.
- **Alleviating Gradient Problems:** In deep networks, gradients can become very small (vanishing) or very large (exploding) as they propagate back through many layers. Residual connections provide a "shortcut" for the gradient to flow directly back to earlier layers- which helps prevent vanishing gradients and stabilizes training.
- **Identity Function Initialization:** At initialization, when weights are small, the layer's contribution is minimal. Adding the input makes the combined output resemble the identity function, which can be a good starting point for learning.
- You can think of it as the network learning a "residual" (the difference) from the identity mapping, rather than learning the entire transformation from scratch.
![[Pasted image 20250918175300.png]]

#### Layer normalization
- Helps model train faster.
- For **layer normalization**, you calculate the mean and standard deviation by **averaging the values across all the dimensions (features) of a single word vector**.
- Then, each dimension of that specific word vector is adjusted using this calculated mean and standard deviation. This process is done independently for _every single word vector_ in your sequence and batch
- Without normalization, the values within a word's vector might vary wildly (some very large, some very small) in different layers or at different points in training. This "uninformative variation" can make it harder for the next layer to learn effectively
- Layer normalization ensures that the input to the subsequent layer always has a consistent scale, making the learning process more predictable and efficient
- Each individual word vector (across its dimensions) to have a unit mean (average of 0) and unit standard deviation (standard deviation of 1)

#### Cross attention
In this the queries are from the input vectors in the decoder whereas the key and value is from the output vector in the encoder. You simply apply the key matrix and the value matrix to the output of the encoder and the query matrix to the input of the decoder. 

#### Quadratic computer problem
Self attention computation is quadratic. This means our attention can't be on a bigger part of the document. 
