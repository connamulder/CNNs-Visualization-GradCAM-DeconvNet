# CNNs-Visualization(Grad-CAM/DeconvNet)
#### Implementation of the visualization of classical CNNs, like VGGNet, ResNet50 and GoogleNet.  

- DeconvNet: VGGNet
- Grad-CAM: VGGNet, ResNet50, GoogleNet

No description of the theory of these two algorithms.

Here are some results of these two methods applied in respective CNNs:

***

#####  Grad-CAM

1. Same image in VGG16 and VGG19(all the convolution layers)

   ![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/VGGNet.jpg>)

   

2. Different classes of one image on VGG16(beagle/hound/egypt cat/sheet)

   ![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/Classes.jpg>)

   

3. Modification of the Grad-CAM: change the ReLU function in the last step, instead, only save the negative pixels of the heat map.  Thus, we could get the area that bring negative effect to the right decision.(the negative area of beagle)

   ![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/negative.jpg>)

4. One class image(more distinct effect)

   ![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/one_class.jpg>)

5. Apply the methods below to the output of some kernel of the last convolution layer in VGG16

   ```
   cv2.applyColorMap()
   ```

   ![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/kernel.jpg>)

***

##### DeconvNet

1. Some kernel's mapping of the last convolution layer

![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/DeconvNet.jpg>)

2. 9 maps in each last convolution layer of a block in VGG16(feature from color, shape to part)

![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/lastconv_in_all_block_Dec.jpg>)

3. Comparison of the same kernels' output when using the same pre-trained weights.![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/comparision.jpg>)

4. Application to the InceptionV3.

   ![Image text](<https://raw.githubusercontent.com/Stardust-Zjt/CNNs-Visualization-GradCAM-DeconvNet/master/Image/InceptionV3.jpg>)

***

Reference:

[ 《Visualizing and Understanding Convolutional Networks》](http://arxiv.org/pdf/1311.2901v3.pdf)

[《Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization》](<https://arxiv.org/pdf/1610.02391v1.pdf>)

[Jacobgil/Keras-Grad-CAM](<https://github.com/jacobgil/keras-grad-cam>)

[jalused/Deconvnet-keras](<https://github.com/jalused/Deconvnet-keras>)

