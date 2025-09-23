###### Zero shot learning
The ability of a model to do many tasks without any training or gradient updates.
###### Few shot learning
specify a task by simply prepending examples of the task before your example. In context learning as there are no gradient updates performed when learning a new task. Just by giving a few examples of a task before asking the model to do the same task, we can achieve the same performance as the state of the art fine tuned model. 
###### Chain of thought prompting
Instead of just showing an example problem and how to solve it, show it the example problem with the steps to solve the problem and the model will perform better. Another way is to give it the problem and then says "let's think step by step." This also forces the model to reason about the problem and try to solve it. 
###### Scaling up finetuning
Earlier, after pretraining, we used to finetune the model on a specific task but now we are going to finetune the model on many tasks. This is done so that users don't have to trick the model into doing their specific task. Instead, users can straight up ask for a specific task to be done. 
###### Instruction finetuning
Collect examples of instruction output pairs across many tasks to finetune a LM. 
Limitations:
- collecting data for instruction finetuning is expensive. For example, you are collecting data to train the model to solve phd level physics questions. 
- tasks like open ended creative generation have no right answer
- language modelling penalizes all token level mistakes the same but some mistakes are worse than others. For example saying that Avatar is a fantasy show is correct. A model could also say that it is a adventure show however, saying that it is a musical is wrong. However, if a model predicts anything but fantasy then these tokens are penalized equally. 
- humans generate suboptimal answers. Model performance is increasing and human beings may not generate good answers against which we compare the models.
###### Optimizing for human preference
Use RLHF - reinforcement learning for human feedback. For an instruction x and a LM sample y, imagine we had a way to obtain a human reward of that summary R(x,y), higher is better. 
Problem 1:  human in the loop is expensive hence we train a separate language model that helps us calculate the reward by judging the LM sample. Example, InstructGPT.
Problem 2: human judgments are noisy and mis calibrated - different people give different scores to the same output. Instead of asking humans to rank an answer ask them to rank different answers. To get a score for the reward model out of these ranks, we use the **Bradley-Terry Model:** The preference data (e.g., "Answer A is better than Answer B") is then used to train the reward model. The Bradley-Terry model is often used, which models the probability of a human preferring one answer over another based on the difference in the internal "rewards" that the reward model assigns to each answer. The model states that the probability that a human chooses answer y1 over y2 is based on the difference between the rewards that humans assign internally and then you take a sigmoid over that. 
###### RLHF pipeline
**1. Pretraining and Supervised Fine-Tuning:**  
The base model, usually a large language model, is pretrained with standard objectives (such as predicting the next token) and optionally fine-tuned on labeled datasets using supervised learning with prompt-response pairs.
**2. Collecting Human Feedback and Reward Model Training:**  
Human annotators review model outputs for a set of prompts, ranking responses or providing comparison data. This feedback is used to train a separate reward model, which learns to predict how well a response aligns with human preferences.
**3. Reinforcement Learning with the Reward Model:**  
The original language model (the “policy”) is further trained using reinforcement learning algorithms (such as Proximal Policy Optimization, PPO). The reward model acts as the reward function, scoring proposed responses to encourage outputs that are rated highly by humans.
###### Problems
The reward model might have errors since it is also a learned outcome and the LM model might learn to generate output that hacks this rewards model by exploiting the policy and we won't achieve the desired optimization. To solve this, we can add a penalty for drifting too far from the initialization.
![[Pasted image 20250922165906.png]]
RLHF can be incredibly complex hence we shift to direct preference optimization.
###### Direct preference optimization
DPO allows for writing the **reward model directly in terms of the language model itself**
Because of this direct relationship, you can **directly fit the reward model to the human preference data** without needing the complex reinforcement learning (RL) step that RLHF traditionally requires. The speaker emphasizes that this is possible because the only true "external information" added to the system throughout the entire process comes from the initial human preference labels.  This implies that if you have those labels, you can directly optimize the language model using DPO without the intermediate reward model and RL steps.

Take the pretrained model and reweight it by the expected reward. Beta is a hyperparameter that governs the tradeoff between the constraint and the reward model. Since the numerator in the closed form solution itself is not a probability distribution we have the Z(x) function to create a distribution by normalizing the numerator. 
![[Pasted image 20250922202830.png]]
![[Pasted image 20250922202924.png]]This equation intuitively makes sense as it tries to take a ratio between the pretrained model and the rewards model. The major issue with Z(X) is that it's **intractable to compute**. This is because it would require summing over _every single possible completion_ for an instruction, not just syntactically correct ones. Given the vast number of tokens (e.g., 50,000+) and arbitrary length completions, this space is astronomically large and computationally impossible to calculate directly. The above can be rewritten as follows:
![[Pasted image 20250922203436.png]]
This essentially states that the current model without any reinforcement learning training still represents a policy and a set of rewards. Hence we wrote it as p RL instead of p * as it is not trained. 
Despite the intractability of the partition function, we can still use it because it cancels out. 
![[Pasted image 20250922203956.png]]![[Pasted image 20250922204005.png]]
