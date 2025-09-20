#### Byte pair encoding algorithm
If at testing time we see a word that is not a part of our vocabulary then we don't know what to do as we don't have an embedding for it. Hence, today we learn a vocabulary of parts of words or subword tokens. So each word is split into a sequence of known subwords. 
The algorithm:
- Start with a vocabulary containing only characters and an end of word symbol.
- Using a corpus of text, find the most common adjacent characters. For example, if a and b are commonly occurring adjacent characters, then add ab as a subword to the vocabulary. 
- Repeat until you reach a desired vocabulary size. 
Example: 
Imagine you have the following text: "banana bandanna bandana"
**Initial Vocabulary (Characters):** {a, b, d, n}
**Step 1: Find the most frequent adjacent pair.**
- "b", "a" appears in "banana", "bandanna", "bandana"
- "a", "n" appears in "banana", "bandanna", "bandana"
- "n", "a" appears in "banana", "bandanna", "bandana"
- "n", "d" appears in "bandanna", "bandana"
- "d", "a" appears in "bandanna", "bandana"
The pair **"an"** appears most frequently (6 times).
**Merge 1: Create new subword "an"**
- **New Vocabulary:** {a, b, d, n, **an**}
- **Text becomes:** "b(an)(an)a b(an)d(an)(an)a b(an)d(an)a" (I'm using parentheses for clarity, these are now single tokens)
**Step 2: Find the most frequent adjacent pair again.**
- Now, the pair **"na"** (from n and a) is very frequent after the first merge (6 times). Let's pick this one.
**Merge 2: Create new subword "na"**
- **New Vocabulary:** {a, b, d, n, an, **na**}
- **Text becomes:** "b(ana)(na) b(an)d(ana) b(an)d(an)a"
Let's continue one more time.
**Step 3: Find the most frequent adjacent pair.**
- The pair **"ana"** now appears frequently (4 times) as a result of previous merges.
**Merge 3: Create new subword "ana"**
- **New Vocabulary:** {a, b, d, n, an, na, **ana**}
- **Text becomes:** "b(ana)(na) b(ana)d(ana) b(an)d(an)a" (Notice "banana" now breaks into "b" + "ana" + "na")

#### Pretraining
Previously, the approach involved:
1. **Pre-trained word embeddings:** Models like Word2Vec would learn a single vector representation for each word, regardless of its context. This means a word like "record" would have one embedding, even though it can be a verb ("to record") or a noun ("a vinyl record"). 
2. **Separate, un-pre-trained networks:** A big box on top, like a Transformer or LSTM, would then be built and its parameters randomly initialized. This network was trained from scratch on specific tasks like sentiment analysis or machine translation, using the pre-trained word embeddings as input.
3. The modern approach, which the lecture focuses on, is to **pre-train the _entire_ network (including the Transformer) simultaneously**. Instead of just learning word embeddings, the whole structure learns deeply about language from vast amounts of unlabeled text.
4. The entire Transformer and its word embeddings are trained together on a very general task (like predicting masked words).
5. This gives the model strong initial parameters and allows it to learn contextual representations, meaning "record" could have different representations depending on its usage in a sentence.
6. These pre-trained models can then be "fine-tuned" with much smaller labeled datasets for specific downstream tasks, leading to significantly better performance than training from scratch.

#### Pretraining encoders
To pretrain encoders we cannot use language modelling as encoders get bidirectional context whereas in language modelling we are trying to predict the next word given the words seen up till now which means there is unidirectional context. 
The solution is to replace some fraction of words in the input with a special [MASK] token and then predict these masked words. 
If x tilda is the masked version of x then we are learning the parameter theta by predicting x given x tilda. Here, we only count the loss terms from words that are masked out. 

#### Pretraining for BERT
Bidirectional encoder representation from transformers
![[Pasted image 20250920105639.png]]
###### Limitations of pretrained encoders
If task involves generating sequences use a pretrained decoder instead. Pretrained encoders don't naturally lead to nice autoregressive (1 word at a time) generation methods. 
###### Extensions of BERT
Mask contiguous spans of words makes a harder, more useful pretraining task - SpanBERT
![[Pasted image 20250920111817.png]]
![[Pasted image 20250920111748.png]]

#### Full finetuning vs parameter efficient finetuning
So far, pretrain and fine tune all parameters. Lightweight fine turning - keep most parameters fixed and only train a few. The intuition is that the pretrained weights were already good and generalized so you keep those. 
###### Prefix-tuning, prompt tuning
Prefix tuning is a technique used in parameter-efficient fine-tuning, which is a method to adapt large pre-trained models to new tasks without having to update all of their original, massive number of parameters.
Here's how it works:
*Freeze the Main Model*: The vast majority of the pre-trained model's parameters are kept fixed and not updated during fine-tuning.
*Add Pseudo Word Vectors*: Instead of changing the model's core weights, a small set of new, trainable parameters are introduced. These are conceptualized as "fake" or "pseudo word vectors" that are prepended (added to the beginning) of the input sequence.
*Train Only the Prefixes*: Only these newly added "prefix" parameters are trained to optimize for the specific downstream task (e.1.g., sentiment analysis).
The intuition behind prefix tuning is to make only the minimal necessary changes to the highly effective pre-trained model, allowing it to retain much of its original generality while still adapting to the new task. It's also much cheaper computationally because fewer parameters need to be updated.
![[Pasted image 20250920112930.png]]
###### Low rank adaptation
Low-Rank Adaptation, often referred to as **LoRA**, is another method for **parameter-efficient or lightweight fine-tuning** of large pre-trained models.
The core idea is to significantly reduce the number of parameters that need to be trained when adapting a large model to a new task. Instead of modifying all the weights of the original pre-trained model, LoRA works by:
1. **Freezing the Original Weights:** The original, large weight matrices of the pre-trained model are kept fixed.
2. **Adding Small, Low-Rank Matrices:** For each original weight matrix, a small, separate "update" matrix is learned. This update matrix is designed to be "low-rank," meaning it can be represented as the product of two much smaller matrices (e.g., a matrix A and a matrix B). This design drastically reduces the number of new parameters needed.
3. **Applying the Update:** During the forward pass, the output from the original frozen weight matrix is combined with the output from this small, low-rank update matrix.
This approach allows for efficient fine-tuning because only the parameters in these small, low-rank matrices are updated, which is a tiny fraction of the original model's parameters. This makes the process much faster and requires less memory, while still achieving strong performance by essentially learning a small "diff" or adjustment to the original highly capable model.
![[Pasted image 20250920112951.png]]

#### Pretraining encoder-decoders
###### Span corruption
The video discusses a highly effective method for pre-training encoder-decoder models called **span corruption** (or salient span masking), which is used in models like T5.
Here's how it works:
1. **Mask Out Spans:** In the input text, contiguous "spans" (sequences of words) are randomly selected and replaced with a single "mask token." For example, "Thank you for inviting me to your party last week" might become "Thank you [MASK_1] me to your party [MASK_2] week."
2. **Generate Corrupted Spans as Output:** The model is then trained to generate the _original masked spans_ as its output. The output sequence would look like: "[MASK_1] for inviting [MASK_2] last."
This method combines the benefits of both encoder and decoder architectures:
- The **encoder** gets a bi-directional view of the entire (corrupted) input sentence, understanding the context around the masked parts.
- The **decoder** is then tasked with generating the missing spans in sequence, which is a generative task, similar to language modeling.
This approach allows the model to learn to both understand context (encoder) and generate fluent text (decoder) by reconstructing the original input, making it suitable for tasks like machine translation or summarization where you need both an understanding of the input and generation of an output.
#### Pretraining decoders
This is just language modelling.
#### GPT 3
 GPT-3 introduced the concept of **in-context learning**. This means it can perform tasks (like translation, arithmetic, or correcting typos) _without_ any fine-tuning, simply by being given examples and instructions within the input prompt itself.  This "emergent property" was highly surprising and indicated that very large models could learn patterns directly from the input.
#### Chain of thought
**Providing Intermediate Steps:** Instead of just giving examples of questions and their final answers, CoT involves including the _step-by-step reasoning process_ within the prompt's examples. For instance, if you ask a math problem, the example would show how to break down the problem into smaller steps before arriving at the solution.
- **Model Mimics Reasoning:** When given a new question, the model learns to first generate its own sequence of intermediate steps (like a "scratchpad" or "thinking out loud") and then produces the final answer based on those steps.
Unclear what the model capabilities are and what its limitations are.
