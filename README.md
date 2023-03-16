
# SPT-LSA ViT : training ViT for small size Datasets

Here is a non official implementation, in Pytorch, of the paper <a href="https://arxiv.org/abs/2112.13492v1">Vision Transformer for Small-Size Datasets<a/>. 

The configuration has been trained on CIFAR-10 and shows interesting results. 

The main components of the papers are :

The ViT architecture :

![image](https://user-images.githubusercontent.com/42917280/225777072-f8f4324a-2ca8-4f82-9548-4b9c0ba83b47.png)

The Shifted Patch Tokenizer (for increasing the locality inductive bias) :

![image](https://user-images.githubusercontent.com/42917280/225777202-72320ff4-6e92-46d4-8dad-0675c643dab6.png)

The Locality Self-Attention : 

![image](https://user-images.githubusercontent.com/42917280/225777292-f3e4d8f6-b3e3-485e-a6d2-da8e116c3943.png)


These components can be found in the models.py 


### Todo

- [ ] Use register_buffer for the -inf mask in the Locality Self-Attention
- [ ] Use warmup  
- [ ] Visualize Attention layers
- [ ] Track scaling coefficient in attention using TensorBoard 

