#### Mixed precision training
fp32 - 32 bits or 4 bytes of memory per number.  First bit is the sign bit, the bits in green are for the range, in other words, how large or small the number can be and the bits in blue are for the precision or how accurately the number can be represented. 
![[Pasted image 20250926122544.png]]
###### Out of memory error
"Out of memory" (OOM) error, often a CUDA error on GPUs, occurs when the model's parameters, gradients, optimizer state, and activations exceed the available GPU memory.
- **Mixed precision training:** The primary solution is to reduce the memory footprint by representing numbers (parameters, gradients) using fewer bits.
- Switching from **fp32** (4 bytes per number) to **fp16** (2 bytes per number) directly halves the memory usage for parameters and gradients. However, fp16 has a smaller "range" (cannot represent very small or very large numbers accurately, leading to values rounding to zero or NaN) and less "precision" (rounding errors).
- **Gradient Scaling:** To address the range issue with fp16, the loss is scaled by a large factor before computing gradients. This "shifts" small gradient values into a representable range for fp16, preventing them from rounding to zero. After the gradients are computed and potentially upcasted to fp32 for master weight updates, the scaling factor is divided out. This helps maintain gradient information for accurate updates.
![[Pasted image 20250926124416.png]]
- Problem with gradient scaling:
- **Manual Tuning:** You have to choose a scaling factor (e.g., 10,000 or 1,000) that works well for your specific model and training dynamics.
- **Dynamic Adjustment:** If the chosen scaling factor is too large, it can lead to gradients becoming `NaN` (Not a Number) values, which can destabilize training. If it's too small, you might still lose precision by rounding small gradients to zero. This means you might need to adjust the scaler dynamically during training based on how the network behaves.
###### bfloat 16
sacrifices precision but keeps the exponent the same. Helps with faster training on a single GPU.
#### Multi-GPU Training
######  Distributed Data Parallel
 In DDP, each GPU processes a different slice of the dataset. After each GPU runs a forward pass and then a backward pass, they will all have computed different gradients based on their unique data slice. To ensure that all copies of the model on different GPUs remain synchronized and learn effectively, these individual gradients need to be combined. The **all-reduce** operation serves this purpose: it synchronizes these gradients across all workers. The all-reduce operation takes pieces of information (in this case, gradients) from all participating GPUs. It performs a reduction operation (like summation or averaging) on these pieces and then distributes the _reduced_ result to _all_ GPUs. - So, after an all-reduce, every GPU will have the full, synchronized (e.g., summed or averaged) gradient from the entire batch of data processed across all GPUs.
**Communication Overhead:** The video states that the communication overhead for this operation is 2 bytes per parameter as gradients are typically in fp16.
###### Poor memory scaling
Naive **Distributed Data Parallel (DDP)** has poor memory scaling because **every single GPU stores a complete and redundant copy of all the necessary training components**.
- **Full Model Parameters:** Each GPU maintains its own full copy of the entire neural network's parameters (weights and biases). For mixed precision training, this might be in **fp16** (2 bytes per parameter).
- **Full Gradients:** After each forward and backward pass, every GPU computes and stores the full gradients for _all_ model parameters. These gradients are also typically stored in **fp16** (2 bytes per parameter).
- **Full Optimizer State:** For optimizers like Adam, each GPU needs to store additional state variables for _every single parameter_ in the model. For example, Adam requires momentum and variance terms.
 If you have a very large model, even if you add more GPUs, each individual GPU still needs to hold all this data, quickly leading to "out of memory" errors on single GPUs despite having multiple. The memory doesn't scale well with the number of GPUs because there's no sharding or distribution of these core components among them.
###### Zero redundancy optimizer
-  Stage 1: Sharding the optimizer state
	Instead of every GPU holding the complete optimizer state for _all_ parameters, the optimizer state is divided into `N` chunks (shards). Each GPU is then assigned to store and manage only **one specific shard** of the optimizer state.
- *Reduce scatter*:  Instead of performing a full `all-reduce` (which would sum gradients for all parameters on all GPUs), a `reduced-scatter` operation is used. This operation:
1. Collects gradients from all GPUs.
2. Sums (or averages) the gradients for each parameter.
3. _Scatters_ only the relevant, combined gradient chunk back to the GPU that is responsible for updating that specific parameter's shard.
So, GPU1 receives the combined gradients only for parameters 1-2 (its shard), GPU2 for 3-4, etc.
- **All-Gather:** After each GPU has updated its specific parameter shard, an `all-gather` operation is performed. This collects all the updated parameter shards from all GPUs and distributes the full, synchronized set of updated parameters back to _every_ GPU. This ensures that every GPU has an identical and up-to-date copy of the complete model before the next forward pass.
![[Pasted image 20250926141310.png]]
- Stage 2 - During the backward pass, gradients are computed layer by layer, starting from the output layer and working backward. The moment the gradients for a specific layer (or group of parameters) are computed on a GPU's data slice, that gradient information is immediately sent to the GPU that is responsible for updating the parameters of that specific layer. This is often called a "reduce" operation. Once the gradient information has been sent, the memory used to store that temporary gradient is immediately deallocated or "destroyed" on the sender GPU. This prevents any single GPU from accumulating all the gradients for the entire model. The designated GPU that received the combined gradient for its responsible layer then uses this gradient, along with its sharded optimizer state, to update its specific shard of the model parameters. Similar to ZeRO Stage 1, after all the parameter shards have been updated across the GPUs, an **all-gather** operation is performed. This collects all the updated parameter shards and distributes the full, synchronized set of model parameters back to every GPU. This ensures consistency for the next forward pass.
- Stage 3 - Full FSDP - - The neural network model is conceptually divided into smaller chunks called **FSDP Units**. These units typically correspond to one or more layers or sub-modules of the model.
- These FSDP units are then converted into "flat parameters," which are essentially contiguous blocks of memory containing the parameters of that unit.
- These flat parameters are then distributed across the GPUs. For example, if you have a model with 100 parameters and 4 GPUs, each GPU might be assigned 25 parameters. Each GPU only stores its assigned subset of the model's parameters. - **Need for Full Parameters:** Since no single GPU holds the entire model, when a GPU needs to process a particular layer or FSDP unit, it must first gather the necessary parameters for that unit.
- **All-Gather Operation:** As a GPU reaches an FSDP unit in the forward pass (e.g., Layer 4), it initiates an **all-gather** operation. This operation collects the shards of parameters for that specific unit from all the GPUs that own them and reconstructs the _full parameters for that unit_ on _all_ participating GPUs
- **Computation:** With the full parameters for that unit now available on each GPU, the forward computation for that layer proceeds.
- **Discarding Parameters:** Crucially, once the computation for that FSDP unit is complete and its outputs (activations) are passed to the next unit, the gathered full parameters for that unit are **discarded** (or "freed") from GPU memory (31:04). This intelligent memory management is key to FSDP's efficiency.
- The video shows how these `all_gather` operations can be overlapped with the actual forward computation of the _previous_ layer. This "pre-fetching" helps hide the communication latency.
If a single batch size fits on a single GPU then use zero stage 2 if no then use zero stage 3 but if even that does not fix the OOM error then try parameter efficient finetuning.
#### Parameter efficient fine tuning
###### Low rank adaptation
- **Main Idea of LoRA**: Instead of updating all parameters of a pre-trained weight matrix (W0), LoRA proposes to update only a much smaller number of parameters by representing the update (delta W) as the product of two low-rank matrices, A and B.
- **Low-Rank Matrices**: The core observation is that the "intrinsic rank" of the gradients during fine-tuning is often low. So, the update is represented as `W0 + delta_W = W0 + B * A`, where B is a D x R matrix and A is an R x K matrix. 'R' is the rank, which is much smaller than D or K, meaning fewer parameters are updated.
- **Alpha Term: An `alpha` term is introduced as a scaling factor, allowing a trade-off between the pre-trained knowledge and the new task-specific knowledge.
- **Trainable Parameters**: Only matrices A and B are trainable, while the original pre-trained weights (W0) are frozen.
- **Benefits of LoRA**:
    - As 'R' (rank) increases, LoRA can converge towards full fine-tuning.
    - Reduced inference latency because you only need to add/remove the small A and B matrices for different tasks.
    - Significantly lower storage cost compared to storing full fine-tuned models for each task.

- **What is 'R' (rank)?**: 'R' is the "rank" of the `delta_W` matrix. In simpler terms, it controls the _expressiveness_ or _complexity_ of the changes that LoRA can make to the original weights. A lower rank means the changes are constrained to a smaller, more specific subspace, while a higher rank allows for more varied and complex changes.
- **The convergence**: If 'R' is very small (e.g., 4 or 8, as often used), `delta_W` is a very constrained update. However, if 'R' were to increase to be as large as the original matrix dimensions (e.g., `R = min(D, K)`), then the product `B * A` can theoretically represent _any_ `D x K` matrix. This means that `delta_W` could effectively become any possible change to `W_0`, which is exactly what happens in full fine-tuning. In full fine-tuning, you update _all_ parameters of `W_0` to become `W_new = W_0 + delta_W_full`, where `delta_W_full` is an unconstrained update. By increasing 'R', LoRA allows `delta_W` to approximate `delta_W_full` more closely, thus converging in performance towards full fine-tuning.

- **Full Fine-Tuning Inference**: If you fully fine-tune a large language model for 10 different tasks, you end up with 10 _entire_ copies of the model, each containing billions of parameters. To switch between tasks during inference, you need to load a completely different, very large model into memory, which is slow and memory-intensive.
- **LoRA Inference**: With LoRA, the original pre-trained model `W_0` remains unchanged and is stored only once. For each task, you train a _small_ pair of `(A, B)` matrices.
- **The "Trick" for Inference** (48:01): When you want to perform inference for a specific task, you dynamically add the corresponding `B * A` product to the original `W_0`. So, the effective weight matrix for that task becomes `W_task = W_0 + B_task * A_task`. This addition happens just-in-time.
- **Reduced Latency**: Switching tasks becomes incredibly fast because you don't reload the entire base model. You just swap out the tiny `(A, B)` matrices for the new task and perform a quick matrix multiplication and addition. This means you can serve many different tasks from a single base model without constantly loading massive files, significantly reducing the "cold start" latency and memory footprint when switching tasks.

