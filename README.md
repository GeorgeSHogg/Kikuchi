# Kikuchi Pattern Identification using AI

A project to investigate the use of Convolutional Neural Networks for identifying materials and indexing them using Electron Backscatter Diffraction (EBSD) images.

## Investigation with Nickel

Using simulated 60x60 grayscale images of nickel kikuchi patterns, CNNs were able to identify individual grains within the specimens. The CNN model was trained and validated to index the nickel samples with a misorientation accuracy of about 0.6 degrees on real world data.

Nickel Map Indexed

![alt text](https://github.com/GeorgeSHogg/Kikuchi/blob/main/Ni%20map.png?raw=true "Nickle Map Indexed")

## Investigation with Steel

Similarly, the investigation was extended to steel specimens using 144x144 images. The computer vision model successfully differentiated between austenite and ferrite phases and indexed them with a misorientation accuracy of about 0.3 degrees. The clear distinction between these phases and the accurate indexing demonstrates the model's potential in materials science applications.

Steel Map Indexed

![alt text](https://github.com/GeorgeSHogg/Kikuchi/blob/main/Steel%20map.png?raw=true "Nickle Map Indexed")

Average neighbour misorientation within grains

![alt text](https://github.com/GeorgeSHogg/Kikuchi/blob/main/Steel%20misorientation.png?raw=true "Nickle Map Indexed")

## Model Architecture

The CNN model architecture was based on resnet50 with a custom head. The model was trained using a simulated dataset of EBSD images, generated using kikuchipy, and optimized to achieve the lowest possible l1 loss and using accuracy as a metric.

## Future Work

- Obtain more data on duplex steels to further understanding.
- Expand the investigation to other materials and image sizes to improve the model's performance.

## Conclusion

This project demonstrates the potential of using CNNs, in identifying and indexing Kikuchi patterns in EBSD images for crystal orientation. The successful indexing of nickel and steel, and differentiation between austenite and ferrite phases in steel, leads to new opportunities to apply these techniques within materials science.
