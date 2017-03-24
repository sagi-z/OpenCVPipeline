OpenCVPipeline - simple show case of using OpenCV with TBB
==========================================================

[For more details see my blog](https://www.theimpossiblecode.com/blog/faster-opencv-smiles-tbb "the impossible code")

How much faster is it?
----------------------
This example executes 30% fatser on my PC. See the blog link above for details.

## Linux Building
### Build dependencies
Make sure these are installed first:
* **cmake** - should already be installed since you built OpenCV by yourself.
* **git** (optional) - you can also download the sources from GitHub.
* **tbb** - you can also download the sources from GitHub.
* **opencv 3.1 and above**

### build
```
git clone https://github.com/sagi-z/OpenCVPipeline.git
cd OpenCVPipeline
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
[Example execution](https://www.youtube.com/watch?v=WUzR5927Mj4 "se it in work")
