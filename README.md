# Unraveling-the-power-of-4D-residual-blocks-in-violence-detection
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
    </li>
    <li><a href="#requirements">Requirements</a>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- Abstract -->
## About the project

In recent years, action recognition has seen significant advancements in using Convolutions Neural Netwrosk models for video analysis. One of the essential fields in this area is violence detection, which determines whether or not  are violent scenes using videos from surveillance cameras. One popular approach to handle this is the Flow Gated Network, two separate networks that extract the features from the frames and the optical flow of the videos using convolutions. However, it cannot capture the spatiotemporal characteristics of the video, which are crucial for accurate action recognition. To address this limitation, researchers have proposed using 4D convolutions at the video level (V4D) and stream buffer in the case of MoViNets. These networks are designed to preserve the 3D spatiotemporal representation of the video while also incorporating residual connections, which allow for better feature propagation and improved performance. In this work, we propose using 4D residual blocks and MoViNets for violence detection on the dataset RFW-2000 to achieve state-of-the-art results in action recognition. This approach compares the strengths of MoViNet and V4D, resulting in more robust and used models for violence detection.

This project was developed by Mike Bermeo student at [Yachay Tech University](https://www.yachaytech.edu.ec/en/).

<p align="right">(<a href="#top">back to top</a>)</p>



### Requirements
* [Python](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/install?hl=es-419)
* [CV2](https://pypi.org/project/opencv-python/)
* [numpy](https://numpy.org/)
* [tqdm](https://pypi.org/project/tqdm/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Dataset
### RWF2000 - A Large Scale Video Database for Violence Detection
* [RWF2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) repository.
* Download the dataset [here](https://duke.box.com/s/kfgnl5bfy7w75cngopskms8kbh5w1mvu)
### Pre-processing
First you need to pre process the dataset to get the optical flows. In order to do that, use the file obtain_opt.py
### Code
You will find in the folder code the original model "flow_gated_network.py", the variation with 8 4D residual blocks "FGN_8_4D.py", the 3D ResNet-18 with 4D residual block "i3d_resnet.py", and the MoViNet models to use it using transfer-learning in the a notebook file "movineta0.ipynb"

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Mike Bermeo - [LinkedIn](https://www.linkedin.com/in/mike-bermeo-1a8869128/) - mike.bermeo@yachaytech.edu.ec

<br>
<br>

Project link: [https://github.com/mikebermeo99/Unraveling-the-power-of-4D-residual-blocks-in-violence-detection](hhttps://github.com/mikebermeo99/Unraveling-the-power-of-4D-residual-blocks-in-violence-detection.git)

<p align="right">(<a href="#top">back to top</a>)</p>
