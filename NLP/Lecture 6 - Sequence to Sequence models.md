#### Perplexity
- **Way to evaluate language model.** Take the probability of a word predicted by a language model and invert it. Do this for each word generated, multiply them together and take geometric average. 
- Another way of calculating this is to calculate the **exponential of the cross entropy loss.** 
- Another way of thinking about this is *"how many uniform choices you're choosing between."* For example, if you have a perplexity number of **64** then you have a **1/64** chance of getting the right answer. 
- **A lower perplexity is better.**
- LSTMs have a perplexity of **30-40.**
- Modern models have **single digit perplexity.**

#### Vanishing Gradients
In RNNs, if the non linearity function is assumed to be the identity function and we calculate the derivate of hidden layer t with respect to hidden layer t - 1 then the partial derivative will be the **weights matrix.** When we compute this gradient for multiple hidden layers, each time the gradient will be the weight matrix and hence we get **powers of the weight matrix** as a gradient. Now if the weight matrix has small numbers then we will have a **vanishing gradient** as we keep on multiplying a small number with itself which reduces the final product. 
Another way of expressing this is as follows:
If the **eigenvalues of the weight matrix are all less than 1** then we will have a vanishing gradient.
The problem is that:
The gradient is being updated based mostly on **nearby vectors** as not as much based on **far away vectors.** This is bad as we need **longer context.** 
In practice, a simple RNN will only condition **~7 tokens back.** Things behind that don't make it to the decision making process. 

#### Exploding gradient
If gradient is too big then the **SGD update becomes too big.** This leads to **bad updates** as we take a large step and reach a weird parameter configuration with a **large loss.** 

###### Gradient clipping
If the **norm of the gradient is greater than some threshold,** scale it down before applying SGD to solve exploding gradient. Norm could refer to the **L2 norm** where we take the square root of the sum of squares of all the gradients components and then **divide the gradient by that.**

![[Pasted image 20250914210351.png]]

#### LSTM
On step t, there is a **hidden state h** and a **cell state c.** Both are vectors of length *n*, the cell stores **long term information.** The LSTM can **read, erase, and write information** from the cell. The cell becomes like *RAM* on the computer. 
The selection of which information is **erased, written or read** is controlled by three corresponding **gates.** Each gate is a probability that is calculated. The gates are also vectors of length *n.* On each timestep, each element of the gates can be **open, closed, or in between.** These gates are **dynamic.** Their value is computed based on **current context.** 

Three gates:
- **Forget gate:** controls what is kept vs forgotten from previous cell state
- **Input gate:** controls what parts of the new cell content are written to cell
- **Output gate:** controls what parts of cell are output to hidden state
- **New cell content:** this is the new content to be written to cell
- **Cell state:** erase some content from last cell state and write some new cell content
- **Hidden state:** read some content from the cell  

**Hidden state** is doing multiple things  
- One part is, feed it into the **output, predict the next token**  
- Second, **store information about the past** for future use  

![[Pasted image 20250914210318.png]]

![[Pasted image 20250915074329.png]]
The **addition** between the **new cell content** and the **old cell state** is the key to **preserving information.** Earlier, we used to **multiply** old and new weights. Having an **addition prevents vanishing gradient.**  
The **forget gate** is more like a *remember gate* as a **1** on the forget gate means you remember everything from the past and a **0** means you forget everything.  

#### Is vanishing/exploding gradient just an RNN problem?
No, it is a problem for **all neural networks, especially deep ones.** Due to **chain rule,** gradient can be **vanishingly small** as it backpropagates. Thus, **lower layers are learned slowly.**

###### Alternate solution for vanishing gradient
**Residual connections** aka *"resnet"* makes deep networks easier to train by having an **identity connection** that preserves information.
![[Pasted image 20250915080314.png]]

#### Other uses of RNNs
- **Part of speech tagging:** *named entity recognition,* identifying which word is a noun, verb etc.  
- **Sentiment analysis:** run an **LSTM** over a sentence and use the **final hidden state** as the **sentence encoding.** Have a **classification layer** that

#### Bidirectional and multi layer RNNs
![[Pasted image 20250915081605.png]]
For bidirectional, run a forward RNN and a backward RNN and concatenate the hidden states together. 
Not applicable to language modeling as we only have left context available. 

###### Multi layer RNNs
Here, the hidden states from RNN layer i are inputs to RNN layer i + 1.
![[Pasted image 20250915082939.png]]

#### Neural machine translation
In 2014, we from from statistical to neural machine translation.
1. **Encoder-Decoder Model:** NMT uses a **sequence-to-sequence model** consisting of two main parts: an encoder and a decoder (1:06:03).
2. **Encoder:** An RNN (specifically an LSTM) processes the source sentence (e.g., French) chunk by chunk. It _doesn't output anything_ during this phase, but builds up a hidden state that encapsulates the entire source sentence's information (1:06:29). The final hidden state of the encoder becomes the starting point for the decoder.
3. **Decoder:** This is another RNN (an LSTM, with different parameters optimized for the target language) that takes the encoder's final hidden state as its initial "memory" (1:06:56). It then generates the translation word by word, using its previous hidden state and the word it just generated to predict the next word (1:07:37).
4. **End-to-End Training:** The entire system (both encoder and decoder) is trained simultaneously using parallel text (sentences and their translations). Losses are calculated for each predicted word in the translation, and these errors are back-propagated through the _entire network_ to update all parameters of both the encoder and decoder (1:08:37). This end-to-end learning aligns all learning for the final translation task.

