#### Developing a machine learning model
- training - minimizing loss
- development - hyperparameter tuning or early stopping, changing learning rate
- model selection - comparing models
- deploying models - is the model good enough for production
#### SuperGLUE
This is a multitask benchmark designed to measure general language capabilities. It's not a single metric itself but a collection of tasks that are evaluated using their own specific metrics. 
#### Evaluating LLMs
Let's break down the main ways researchers evaluate LLMs like GPT-4, and the challenges they face. There are three big categories:
1. **Embeddings-Based Metrics**: These use models like BERT to measure how similar two texts are, based on their meaning—not just exact words. Examples: BERTScore, BLURT. But, they still depend on having good human-written reference answers.
2. **Reference-Free Evaluation**: Instead of comparing to human answers, we let LLMs themselves judge outputs. Tools like AlpacaEval and MT-bench use models as evaluators. Sometimes, LLMs agree with humans more than humans agree with each other!
3. **Human Evaluation**: Humans rate outputs directly. This is the "gold standard" for open-ended tasks, but it's slow, expensive, and people often disagree or make mistakes.

