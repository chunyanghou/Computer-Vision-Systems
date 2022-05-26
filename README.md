# Computer-Vision-Systems
Homework
Tasks
Using the datasets provided, you have to perform the following tasks:
1. 2D Object Detection:
This task can be done with or without neural networks. You simply have to detect all 5 object types, provide accurate bounding boxes and class labels. You can initially use a pre-trained neural net, but for really great results it is recommended that you fine-tune using your own data (don't forget to do a train-validation split). You can also try and enhance the network performance by adding depth as an extra channel (note: this is difficult).
2. 3D Object Detection:
This tasks has to be done without neural nets. Use the results of the 2D tasks (if not completed yet, you can use the correct outputs in the meanwhile) to compute the object locations in 3D relative to the car. Also try and track the objects on multiple frames and try to determine their velocity.
3. Semantic Segmentation:
Use a neural network to perform semantic segmentation on the RGB images. You can initially use a pre-trained network, but it is also recommended that you fine-tune using your own data, just like in the first task. Similarly, you can add the depth image as an extra feature either as an input channel, or as an input to the last layer (note: this is considerably easier to do here than in the first case).
4. Freespace Segmentation:
Use traditional methods (meaning you can't use neural nets here) to find pixels belonging to the road. Note, that the car is always in a fixed position and orientation relative to the ground plane, so this task is relatively easy to do using simply 3D geometry (simplified RANSAC-like methods).
