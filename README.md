# PatchMatch-Unity
GPU implementation of Patch Match in Unity using Compute Shader

<img src="./PatchMatchUnitySample.png" width="50%">

## License
- MIT License

## Tested Environment
- Unity 2018.3.7f1
- Windows 10 Pro (Version: 1803, OS build: 17134.345)
- NVIDIA GeForce GTX 970

## References
- PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing (2009)
    - [Project Page](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php)
    - [Paper](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)
    - [Source Code](http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/patchmatch-2.1.zip)
        - pm_minimal.cpp
        - nn.cpp

- PatchMatch GPU implementation using CUDA
    - https://github.com/rozentill/PatchMatch
    - https://github.com/rozentill/PatchMatch/blob/master/src/PatchMatch-GPU/PatchMatch-GPU/kernel.cu

- 参考書庫 sanko-shoko.net
    - [PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing
](http://www.sanko-shoko.net/note.php?id=jnzb)
