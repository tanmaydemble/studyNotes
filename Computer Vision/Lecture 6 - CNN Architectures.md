#### Normalization
- **Per-Sample Normalization**: Unlike some other normalization methods, LayerNorm calculates the mean and standard deviation for _each individual sample_ (or input) across all its features or dimensions.
- **Two Steps**:
    1. First, it normalizes the input data to have a mean of zero and a standard deviation of one (unit Gaussian).
    2. Then, it applies learnable parameters (gamma for scaling and beta for shifting) to transform this normalized data. These parameters allow the model to learn the optimal distribution for the data at that point in the network.
In the context of CNNs, for an image with channels, height, and width, LayerNorm computes one mean and one standard deviation across _all_ channels, heights, and widths for each individual image in a batch. This makes it different from BatchNorm, which normalizes across the batch dimension for each channel.
The main purpose of LayerNorm (and other normalization layers) is to stabilize the training process, allowing for deeper networks and faster convergence.
![[Pasted image 20250929181437.png]]
- **Layer Normalization (LayerNorm)**: Normalizes features **across all channels and spatial dimensions for each individual sample** in a batch (6:51).
- **Batch Normalization (BatchNorm)**: Normalizes features **across the entire batch and spatial dimensions for each individual channel** (7:35).
- **Instance Normalization (InstanceNorm)**: Normalizes features **across spatial dimensions for each individual sample and each individual channel** (7:46).
- **Group Normalization (GroupNorm)**: Normalizes features **across a predefined group of channels for each individual sample**, while also considering spatial dimensions (6:30).
Imagine you have a big pile of photos, and you want to make sure each photo has good lighting and color.
**LayerNorm** is like adjusting the brightness and contrast for _each single photo individually_. It looks at all the pixels in _that one photo_ and makes sure its colors are balanced and not too dark or too bright, regardless of what the other photos in the pile look like.
This helps the computer model learn better because every photo it sees is consistently "well-lit" within itself, making it easier to find patterns.
#### Dropout
- **Core Idea**: Dropout adds randomness during the training process by randomly setting some of the outputs (or "activations") of a layer to zero during each forward pass (10:04).
- **Hyperparameter**: The main parameter is the **probability of dropping out values (p)**, often set to 0.5 or 0.25 (10:25).
- **Why it Works (Intuition)**:
    - It forces the network to learn **redundant representations** (11:22). If certain features are randomly dropped out, the model cannot over-rely on any single feature or combination of features always being present.
    - This encourages the model to learn a more **broad and robust set of correspondences** between features and output classes, helping it generalize better to new, unseen data (11:46).
- **Training vs. Test Time**:
    - **During Training**: Dropout is active, randomly zeroing out activations. This makes the model perform _worse_ on the training data, as it's not seeing all the information (13:17).
    - **At Test Time**: Dropout is **turned off** (13:31). All activations are used. However, because more activations are present than during training (e.g., if 50% were dropped, now 100% are there), the outputs would be much larger. To compensate, the outputs are **scaled by the dropout probability 'p'** (e.g., multiplied by 0.5 if p=0.5) to maintain the same expected magnitude as during training (13:50). This "inverted dropout" is a common implementation.
- **Backpropagation**: When values are zeroed out by dropout, their corresponding gradients also become zero, meaning the weights associated with those dropped activations are not updated during that specific step.
#### Feature coadaptation
**Feature coadaptation** refers to a situation where, during training, a neural network's neurons become overly reliant on the presence of specific other neurons or features. They might develop strong, interdependent relationships, where one neuron's activation is only meaningful if another specific neuron is also active.
Imagine a group project where each person only knows how to do their part if specific other team members are doing theirs in a very particular way. If one of those key members is absent, the whole project falls apart.
In the context of the video's explanation (around 11:46), the presenter states that dropout prevents the model from "over rely on certain features being present." For example, a model might learn that "ears AND furry" strongly indicates a cat. If these two features are always present together in the training data, the neurons processing them might coadapt, meaning they only fire or contribute meaningfully when both are active.
**How Dropout Prevents Coadaptation:**
Because dropout randomly "blanks out" (zeros out) a percentage of activations in each training step, it breaks these specific co-dependencies. A neuron can no longer reliably assume that its "partner" neuron will always be active.
This forces each neuron (or feature detector) to become more robust and useful _independently_, or to find other, more generalizable combinations of features. It's like forcing each team member to learn how to complete their task even if others are absent, making the whole team more adaptable. This leads to better generalization on unseen data because the model isn't "locked in" to specific, brittle relationships between features learned only from the training set.
#### Activation functions
###### Sigmoid
Main issue: empirically, after many layers, it leads to **smaller and smaller gradients** during backpropagation. The speaker points out that if you look at the graph of the sigmoid function, the gradient is very flat (close to zero) for very negative and very positive input values. This means that for almost all input space, the gradients are very small, making it difficult for the model to learn effectively in those regions.
###### ReLU
The video explains that **ReLU (Rectified Linear Unit)** became very popular because it addresses the vanishing gradient problem seen in sigmoid functions. For positive input values, the derivative is always 1, meaning there's no vanishing gradient in that region. For negative input values, the gradient is 0, which means there's still a "flat portion" where the gradient is zero. However, this is better than sigmoid, where almost the entire input domain can have very small gradients. ReLU is **much cheaper to compute** than sigmoid because it only involves a simple max operation (max(0, input value)). The negative region is a problem which lead to the rise of GELU.
###### GELU
Main activation function used in transformers today. It addresses ReLU's issue of having a zero gradient for any negative input by having a **non-flat section of the activation function near zero**. GELU is designed to **smoothen out the sharp, non-smooth jump** in the derivative from 0 to 1 that occurs at 0 in ReLU. As input values approach positive or negative infinity, GELU **converges to ReLU's behavior**. GELU calculates the **Gaussian Error Linear Unit**, which involves the cumulative distribution function of a Gaussian normal.
#### VGG 
VGG's architecture is remarkably simple, primarily consisting of stacked **3x3 convolution layers** with a stride of 1 and padding of 1, followed by max pooling layers. The final layer of VGG (and similar models) has 1000 outputs because it was trained on ImageNet, which has 1000 different image categories.  A significant discussion around VGG focuses on why stacking multiple 3x3 convolution layers is beneficial. Three stacked 3x3 convolutions with stride 1 have the **same effective receptive field as a single 7x7 layer**. Stacking these smaller filters results in **fewer parameters** compared to a single larger filter and allows the model to learn **more complex, non-linear relationships** due to the multiple activation functions between the stacked layers.
- **First 3x3 layer**: A single neuron in the output of the first 3x3 convolution layer "sees" a 3x3 area of the input image.
- **Second 3x3 layer**: A neuron in the output of the second 3x3 convolution layer "sees" a 3x3 area of the _previous_ layer's output. Since each point in the previous layer saw a 3x3 area, this effectively means the neuron in the second layer's output "sees" a **5x5** area of the original input image. The video states that with stride 1, you're always adding two to your receptive field (26:48).
- **Third 3x3 layer**: Following the pattern, a neuron in the output of the third 3x3 convolution layer "sees" a 3x3 area of the _second_ layer's output. This translates to an effective **7x7** area of the original input image.
1. **Why this is Beneficial (28:03)**:
    - **Fewer Parameters**: A stack of three 3x3 filters requires fewer parameters than one 7x7 filter, assuming the same number of input and output channels. For example, if you have `C` input and output channels, three 3x3 filters would have `3 * (3*3*C*C)` parameters, while one 7x7 filter would have `(7*7*C*C)` parameters. The `27 * C^2` vs `49 * C^2` shows `3x3` is much more efficient.
    - **Increased Non-linearity**: By stacking multiple layers, you introduce **more activation functions** (non-linearities) into the network (28:42). Each activation function allows the model to learn more complex patterns and relationships in the data, making the overall model more expressive and powerful. A single 7x7 layer would only have one activation function for that receptive field.
#### ResNet
**deeper plain CNN models can perform _worse_ than shallower ones, even on training data**. This is counter-intuitive because deeper networks theoretically have more representational power and _should_ be able to learn at least as well as shallower ones. The key takeaway in this segment is that this performance degradation is **not due to overfitting** (30:04) but because **deeper models are harder to optimize**. The larger search space of possible functions makes it more difficult for optimization algorithms (like gradient descent) to find good solutions. The video then leads into the idea of how a deeper model _could_ be at least as good as a shallower one: by setting some layers to be an **identity function** meaning they just pass the input through unchanged. This sets up the introduction of residual connections.
###### Residual connections
- **The Problem**: In a traditional deep network, each block of layers tries to learn a direct mapping from its input `x` to its desired output, let's call it `H(x)`.
- **The Residual Solution**: With a residual connection (or skip connection), the block is designed to learn a _difference_ or _residual_ `F(x)` (32:14). This `F(x)` is then added to the original input `x` to produce the final output of the block. So, the desired output becomes `H(x) = F(x) + x`.
- **Ease of Learning Identity**: This formulation makes it extremely easy for the network to learn an **identity function** for a block (i.e., `H(x) = x`). If the block doesn't need to transform the input at all, it can simply learn `F(x) = 0` (by setting filter weights to zero), and the input `x` will pass through via the skip connection (32:39).
- **Learning Small Differences**: More practically, it means the block only needs to learn the _difference_ between the input and the desired output, rather than the entire complex mapping (33:01). This learning task is generally simpler, helping to avoid optimization issues like getting stuck in poor local minima and enabling the training of much deeper networks.
![[Pasted image 20251001175944.png]]
![[Pasted image 20251001180306.png]]
###### Filter doubling and downsampling
Periodically within the network, ResNets will **double the number of filters** (increasing depth) and **downsample the spatial dimensions** (height and width) of the activation maps (40:38). This leads to a network where activations become spatially smaller but have greater depth as they progress through the layers.
###### Initial large convolution
ResNets often start with a relatively larger convolution layer (e.g., 7x7) before the residual blocks, an empirically found design choice. 
#### Weight initialization
 If weights are initialized to values that are **too small**, the activations (outputs) of each subsequent layer can shrink towards zero, leading to a "mode collapse to zero".  This makes it difficult for gradients to flow backward during training, hindering learning (vanishing gradient). Conversely, if weights are initialized **too large**, activations can explode, growing increasingly large with each layer (43:40), leading to unstable training (exploding gradient).  Ideally, you want the **mean and standard deviation of activations to remain relatively constant** across all layers of the network during the forward pass (43:18). This ensures that information can flow effectively through the network and the optimization problem is easier to solve.  
###### Kaiming initialization
 Kaiming initialization is a method for initializing the weight values of layers in a neural network. It was developed by Kaiming Hu, who is also known for creating ResNets. The main idea behind Kaiming initialization is to set the initial weights based on the input dimension size. 
###### Data Augmentation
Flipping the images - not good for models that read texts though - do a vertical flip. 
Resizing and cropping - take a random crop of the image and resize that to be the normal size of your image. 
Test time augmentation - Average a fixed set of crops - like take an image, crop it and resize it or flip it and average out the model output on the same image.
Color jitter- change the contrast or brightness of the image to help generalize.
Cutout or random crop - black out or cover one part of your image and then use that to train the model
#### Transfer Learning
If you don't have a lot of data you can still train CNNs. 
![[Pasted image 20251001210051.png]]
Essentially, freeze the model that was trained on image net and then just retrain the final classifier to predict the classes you want. This should work because the frozen model learns to find edges and different colors in images and that sort of recognition should remain same across images no matter what sort of classification is to do done. Do this if you have a small dataset. However, if you have a bigger dataset then initialize the values of the model to the ones trained on imagenet but then finetune all the model parameters, even the ones we froze for the smaller dataset to solve the new classification problem. 
However, if you have a very different dataset and you have a lot of data you can start from scratch or if you have less data then find a model trained on something closer. 
#### Hyperparameter
- **Overfit on a Small Sample for Debugging:** The first step for debugging is to ensure your model can overfit a small sample of data, even just one data point (1:07:22). If it can't, it indicates a bug or an unsuitable model architecture. This also helps in finding a reasonable range for learning rates.
- **Coarse Grid Search (Initially for Learning Rate):** After debugging with a small sample, you can try a coarse grid search, starting with different learning rates to see which ones lead to the most sustained decrease in training loss
- You should monitor both training and validation accuracy and loss curves. If validation loss goes up while training loss goes down, it indicates overfitting, requiring more regularization or data (1:08:39). If there's little gap between training and validation accuracy, you can usually continue training longer
- The video strongly recommends using **random search** over a grid search for hyperparameter optimization (1:09:40). Random search is more efficient because it explores the hyperparameter space more thoroughly, especially for important hyperparameters, by randomly sampling values within defined ranges.
###### If training loss and validation loss gap increases - overfitting
- **Training Loss Decreases (or stays low):** The model continues to learn and fit the training data very well, leading to a low or decreasing training loss. It's essentially memorizing the training examples (1:08:49).
- **Validation Loss Increases (or plateaus):** However, this "memorization" doesn't generalize to new, unseen data (the validation set). The model performs poorly on data it hasn't seen before, causing the validation loss to increase or stop improving.
###### If training loss and validation loss gap decreases - underfitting
1. **Model Not Learning Enough:** If both losses are high, it means the model isn't performing well even on the data it has seen (training data), and similarly, it's not performing well on new data (validation data).
2. **Too Simple or Not Trained Enough:** This indicates that the model might be too simple to capture the complexity of the data, or it hasn't been trained for a sufficient number of epochs to learn the underlying patterns effectively. The model hasn't learned the fundamental relationships from the data yet.
###### Grid search vs random search
why random search is generally better than grid search for hyperparameter tuning:
- **More Efficient Exploration:** Random search explores the hyperparameter space more efficiently, especially when some hyperparameters have a much greater impact on performance than others (1:09:49).
- **Better Coverage of Important Parameters:** Instead of testing only fixed points for each parameter (like in a grid), random search samples widely, increasing the chance of finding better values for the truly influential hyperparameters (1:10:09).
- **Avoids Artificial Limitations:** Grid search is limited by its rigid, predefined steps, which can cause you to miss optimal values that lie between your chosen grid points. Random search is not constrained by these fixed points.
- **Discovers More Diverse Combinations:** For the same number of trials, random search tends to explore a wider variety of unique hyperparameter combinations compared to grid search.
