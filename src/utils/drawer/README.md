# ConvNet Drawer

Python script for illustrating Convolutional Neural Networks (CNN).
Inspired by the draw_convnet project [1].

Models can be visualized via Keras-like ([Sequential](https://keras.io/models/sequential/)) model definitions.
The result can be saved as SVG file or pptx file!

## Requirements
python-pptx (if you want to save models as pptx)

```sh
pip install python-pptx
```

Keras (if you want to convert Keras sequential model)

```sh
pip install keras
```

matplotlib (if you want to save models via matplotlib)

```bash
pip install matplotlib
```

## References
[1] https://github.com/gwding/draw_convnet

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," in Proc. of NIPS, 2012.