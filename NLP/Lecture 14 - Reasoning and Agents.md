#### Self-consistency
Self-consistency is a prompting technique used with language models to improve their reasoning ability. Instead of generating a single reasoning path and answer, the language model is prompted to:
1. Generate **multiple different reasoning paths (rationals)** to a problem.
2. For each rational, generate a **corresponding answer**.
3. Compare all the generated answers and **select the one that appears most frequently** (the majority vote).