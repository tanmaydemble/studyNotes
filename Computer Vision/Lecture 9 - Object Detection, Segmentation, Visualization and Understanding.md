#### Vision Transformer Image Classification Process
1. **Image Splitting into Patches**
    - The input image is divided into NN patches.
    - Example: Split the image into 9 equal sections.
2. **Patch Embedding**
    - Each sub-image (patch) is flattened and converted into a dd-dimensional vector using a linear projection.
3. **Positional Embeddings**
    - A learnable dd-dimensional positional embedding is added to each patch embedding to retain spatial information.
4. **Transformer Input**
    - The set of patch embeddings (with positional encodings) is fed into the transformer encoder.
5. **Output Processing Approaches**  
    There are two common strategies for converting transformer outputs into a classification input:
    - **Class Token Method**
        - Introduce a learnable special **class token** as an extra input.
        - After transformer layers, the output of the class token is extracted.
        - This token’s output is transformed into the class probability vector.
    - **Pooling Method**
        - Instead of a class token, all patch output vectors are aggregated.
        - Use **Global Average Pooling (GAP)** across patches to combine them into a single vector.
        - GAP summarizes the features by averaging across spatial positions.
6. **Classification Head**
    - The resulting vector (from class token or pooling) is passed through one or more **fully connected (dense) layers** for projection into class logits.
7. **Softmax Activation**
    - A **softmax function** is applied to the logits.
    - This converts raw scores into a probability distribution across predefined classes, ensuring values are between 0 and 1 and sum to 1.
#### Optimizations and Tweaks - To Stabilize Training
###### Layer Normalization
- Earlier the layer normalization block was after the residual connection block. This would prevent the transformer from learning the residual connections. However, now we place a layer norm right before the self attention and residual part and also right before the MLP part.
- **Intuition behind this:** In a standard residual block, you add the input to the output of a sub-layer (e.g., self-attention). If layer normalization is applied _after_ this addition (Post-LN), the feature values are immediately changed by the normalization.
- This means that even if the sub-layer (like self-attention) theoretically learns to output zero, the normalization step would still alter the identity path, making it difficult for the network to truly learn an identity mapping.
- By placing the layer normalization _before_ the self-attention and MLP, the residual connection itself can maintain the original, un-normalized feature values as they pass through, allowing the network to learn a more effective identity transformation if needed. This contributes to **more stable training** and **better performance** for very deep networks.
###### Root Mean Squared Norm
- This norm is used instead of layer norm. **RMS Norm** (Root Mean Square Normalization) is a more basic type of normalization compared to Layer Norm.
- The key difference highlighted is that **RMS Norm does not use the mean value of the features for normalization**. It only scales the activations based on their root mean square, while Layer Norm normalizes by both mean and variance.
- The primary reason for its adoption is that it has been **empirically shown to make training a little bit more stable** for certain architectures and tasks.
###### SwiGLU MLP
- In the context of a transformer block, a standard MLP typically consists of **two linear (or dense) layers** with a non-linear activation function (like ReLU or GELU) applied in between them. You can think of it as: Input → Linear Layer 1 → Activation → Linear Layer 2 → Output. This involves two sets of weights (W1 and W2).
- SwiGLU introduces a **"gated non-linearity"**. Instead of just two weight matrices, it uses **three weight matrices** (W1, W2, and W3).
- The input is split, one part goes through a linear transformation and an activation (often Swish or GELU), and the other part goes through a linear transformation. These two paths are then multiplied element-wise (gated).
- This structure helps in:
    - Creating a **better non-linearity**
    - Learning **higher-dimensional non-linearities**, even when keeping the network size (number of parameters) similar to a classic MLP by adjusting the hidden layer dimension
###### Mixture of Experts
- Instead of having just **one set of MLP (Multi-Layer Perceptron) layers** in a network block, an MoE architecture has **multiple sets of MLP layers**, each referred to as an "expert".
- A **router** component (often a small neural network itself) is introduced. This router takes the input token (or feature) and determines which of these "experts" should process it. The token is then routed to one or more of these experts.
- **Increased Capacity without High Compute**: MoE allows for a significant increase in the total number of parameters in the model (more experts mean more parameters) without a proportional increase in the computational cost _per token_ during inference. This is because each token only activates a subset of the experts, not all of them.
- **More Robust Models**: By having different experts, the model can learn different aspects or "modes" of the data. This allows it to become more robust and potentially cover multiple probability distributions within the data.
#### Common machine learning tasks
- Object classification
- Image segmentation: Each of the pixel should be labelled such that it is a part of an object from the picture. 
- Object detection: Identifying objects and drawing bounding boxes around them along with their class labels. 
- Instance segmentation: Combining object detection with semantic segmentation to not only label pixels but also distinguish between different instances of the same object. 
#### Semantic or image segmentation
Simply looking at individual pixels is insufficient because there's no context. Context from surrounding areas is crucial for accurate labeling. An early idea was to take small patches (a pixel and its surroundings), feed each patch into a Convolutional Neural Network (CNN) to classify the central pixel. However, this method is **extremely time-consuming** as it requires running a full network for every single pixel.  To address the speed issue, the concept of FCNs is introduced. Instead of processing patches, an FCN (Fully convolutional network) takes the **entire image as input** and directly **outputs a full segmentation map** (a matrix of labels).

FCNs need to maintain spatial dimensions throughout the network, meaning they cannot use fully connected layers that flatten the input. The network needs to remain "inflated" to produce an image-sized output.  A caveat with FCNs is that keeping layers large to match image dimensions results in a huge number of parameters, making training computationally expensive, especially with older hardware.

The challenge of having very large layers in early **Fully Convolutional Networks (FCNs)** due to maintaining the original image resolution throughout was solved by evolving the architecture to include **downsampling** and **upsampling** phases.

1. **Downsampling (Encoder Path)**: The network starts with the full-size image but progressively **reduces the spatial resolution** (e.g., width and height) of the feature maps through operations like pooling (e.g., max pooling) or strided convolutions (21:45). This makes the feature maps smaller but often "thicker" in terms of the number of channels, capturing more abstract, high-level features. This "shrinking" reduces the number of parameters and computational load in deeper layers.
2. After processing the low-resolution, high-channel features, the network then needs to **upsample** these features back to the original image resolution (or the desired output segmentation map resolution) (21:55). This involves operations like unpooling, transposed convolutions (also called deconvlutions), or nearest-neighbor upsampling.
This "U-shaped" architecture (like the U-Net, which the video discusses later) allows the network to efficiently learn both high-level contextual information from the downsampled path and fine-grained spatial details from the upsampled path, without the extreme computational cost of very large layers throughout.

Loss function for this network: Essentially, for each pixel, the network performs a classification task to determine its label. Therefore, the total loss is calculated as the **sum (sigma) of the softmax loss for all pixels** across the entire image
###### Unpooling
![[Pasted image 20251003111453.png]]
![[Pasted image 20251003112046.png]]
Transposed convolution: - Instead of convolving the input data directly, it implies doing the convolution on a "transposed version of the input," which effectively generates a **larger output** (31:50). This means it increases the spatial dimensions of the feature map.
#### Instance segmentation
- **Goal**: The task of object detection is to not only classify objects in an image but also to **localize them by predicting their bounding box coordinates** (X, Y, Height, Width) (34:33).
- **Loss Function**: It's typically trained using a **multi-task loss function**. This combines a classification loss (like softmax loss for the object's class) and a regression loss (like L2 loss for the bounding box coordinates) (35:00).
- **Challenge with Multiple Objects**: While simple for a single object, detecting multiple objects in a scene becomes challenging due to the need to generate many output numbers (class scores and box coordinates for each object) (
- **Sliding Window (Inefficient)**: An early, inefficient idea was to use a sliding window approach, classifying every possible bounding box as either an object or background (36:58). This was not scalable due to too many combinations.
- **Region Proposals (R-CNN, Fast R-CNN)**: A significant advancement involved generating "region proposals" – areas likely to contain objects – and then processing only those regions. R-CNN and its faster variants fall into this category (37:54). These methods, however, were often computationally heavy. 
-  R-CNN addresses the problem of detecting multiple objects by first identifying "region proposals" – areas in the image that are likely to contain an object (37:54). For each proposed region:
    - The patch is extracted from the image.
    - A full **Convolutional Neural Network (CNN)** is run on that patch (38:24).
    - This CNN then classifies the object within the patch and refines the bounding box coordinates (38:38).
- **Drawback**: R-CNN is **very slow** because a full CNN has to be run independently for _each_ region proposal in the image (39:09).
- **Fast R-CNN**: This improved version addresses the speed issue by first running **one large CNN on the entire image** to generate a feature map. Then, the region proposals are applied directly to this feature map, and a smaller CNN is run on those regions of the feature map. This avoids redundant computation, making it much faster. 
![[Pasted image 20251003121015.png]]
###### Region proposal network
A **Region Proposal Network (RPN)** is a specialized neural network component designed to automatically generate **region proposals**, which are bounding boxes in an image that are likely to contain an object.
- **Purpose**: Earlier object detection methods like R-CNN relied on external, often slow, algorithms to find these regions. RPNs integrate this step directly into the deep learning pipeline, making the entire detection process more efficient.
- **Process**: An RPN typically starts by applying convolutions to the input image (or its feature map from a backbone CNN). It effectively "looks" at different locations and scales within the image.
- **Learning**: Through these convolutional layers, the RPN learns to **refine initial box guesses** and predict which regions have a **high probability of containing an object** (41:42).
- **Output**: The RPN outputs a set of refined bounding box coordinates and an "objectness score" for each, indicating the likelihood that an object is present (42:24).
- **Selection**: From these generated proposals, only the **top-k** (e.g., the top 2000) regions with the highest objectness scores are selected as the final region proposals to be passed to the next stage of the object detection pipeline
#### YOLO
- It's known for being a **fast object detector** and is very good at detecting objects 
- Even its earlier versions are still used in many **industrial applications** today
- The core idea is to process the image with **one single pass** to generate all bounding boxes
- It divides the image into an S by S grid (e.g., 7x7) 
- For each grid box, a fully convolutional network outputs the **probability of an object being in that location**, bounding box refinements, and class probabilities 
- Multiple bounding boxes with associated probabilities are generated, and then **thresholding and non-maximal suppression** are used to identify the most likely objects
#### DETR - Object detection with transformers
- The image is split into **patches**, which are then converted into **tokens** (similar to Vision Transformers for classification)
- **Positional encoding** is added to these tokens, which are then fed into a **Transformer Encoder
- The encoder's output tokens are passed to a **Transformer Decoder**. Crucially, the decoder also takes "object queries" as input 
- These **object queries** are learnable parameters, and each query asks the model to detect an object. If you input 10 queries, you're seeking up to 10 objects 
- Through self-attention and cross-attention layers within the decoder, these queries interact with the encoder's output.
- Finally, a Feed-Forward Network (FFN) processes the decoder's output for each query to generate the **class label** (or "no object") and the **bounding box coordinates**
- The entire process is supervised using **class probability loss** and **L2 regression loss** for the bounding box coordinates
###### Zero shot learning
Zero shot means understanding something new without having a corresponding example in the training data. 
#### Saliency
**Saliency** is a way to understand and visualize neural networks. It helps determine **which pixels matter** in an image for a specific classification decision. For example, in medical imaging, it's crucial to know _where_ a tumor is, not just if it exists. The simplest way to calculate saliency is by taking the **gradient of the class score with respect to the pixel values** in the input image. This gradient indicates how much changing a pixel's value would affect the network's score for a particular class. Visualizing these gradients highlights the pixels that are most influential for the classification. 
1. **Input:** You have an input image (pixels) and a trained neural network that outputs scores for different classes.
2. **Target Score:** Choose one specific class score you're interested in (e.g., the score for "dog" if the image contains a dog).
3. **Backpropagation:** Instead of backpropagating the loss to update the network's weights (as done during training), you backpropagate this _specific class score_ all the way back to the input pixels.
4. **Gradient Calculation:** This process calculates the **partial derivative of the target class score with respect to each individual pixel value** in the input image.
#### Class Activation Mapping
Used for understanding CNNs. CAMs allow you to **trace back class predictions** to specific spatial locations within the network's feature maps. By multiplying the weights of the final classification layer with the feature maps from the last convolutional layer, you can create a **weighted sum** that highlights areas of the image that are most important for a given class. These maps can then be **upsampled back to the original image space**, showing heatmaps that indicate which regions or pixels activate strongly for specific classes. A limitation of the original CAM algorithm is that it can **only be applied to the last convolutional layer** of the network. 
- The model processes the image through many layers, creating internal "feature maps" that are like simplified versions of the image, highlighting different patterns.
- CAM essentially looks at the _last_ of these pattern maps, just before the model makes its final decision.
- It then figures out which parts of that final pattern map were most active or important when the model decided "cat."
- Finally, it creates a **heatmap** on top of your original image, showing the "hot spots" (often in red or yellow) where the model's attention was highest for that specific class.
###### Grad CAM
- The original CAM could only be applied to the _last_ convolutional layer of a network. This is because its calculation method relied on the specific architecture of having a global average pooling layer right before the classification layer.
- Grad-CAM overcomes this by using **gradients** to calculate the weights for the feature maps.
- For a specific class (e.g., "dog"), you calculate the **gradients of that class's score** with respect to the activations in your chosen target feature maps. These gradients tell you how much each activation in that layer influences the final "dog" score.
- For each feature map in your chosen layer, you average these gradients across all its spatial locations. This average gradient becomes a "weight" that represents how important that entire feature map is for the "dog" class.
 - You then multiply each of your original target feature maps by its corresponding importance weight (calculated in step 3).
 - **Apply ReLU:** A ReLU (Rectified Linear Unit) activation is applied to this combined map. This is crucial because it only keeps the **positive influences**, focusing on features that _increase_ the class score, not decrease it (1:11:30-1:11:32).
- **Upsample and Overlay:** The resulting combined map is then upsampled to the original image size and overlaid as a heatmap. This heatmap visually highlights the regions in the image that were most relevant for the network's prediction of that specific class.
Imagine your convolutional layer has produced several different "pattern maps" (these are your **target feature maps**). Each map highlights a different kind of feature it found in the image, like one map showing all the edges, another showing all the circular shapes, and so on.
Now, in the previous step (Step 3), we calculated an **"importance weight"** for _each one of these individual pattern maps_ for a specific class (like "dog"). This weight tells us how much that _entire pattern map_ contributes to the final "dog" prediction.
The "Weighted Combination" step simply means:
1. Take the first pattern map (e.g., the "edges" map).
2. Multiply _every single value_ in that "edges" map by its calculated "importance weight."
3. Do the same for the second pattern map (e.g., the "circular shapes" map), multiplying _every single value_ in it by _its_ importance weight.
4. Repeat this for _all_ the pattern maps.
Once you've done this multiplication for all the maps, you **sum all these weighted maps together** into one single map. This final combined map now emphasizes the areas and patterns that were most relevant for the "dog" prediction across all the different features the layer detected.
It's like taking multiple transparent sheets, each with a different pattern drawn on it, and then adjusting how transparent or dark each sheet is based on how important that pattern is. When you stack them all up, the most important patterns shine through or appear darkest.