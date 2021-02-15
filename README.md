# SliceNStitch: Continuous CP Decomposition of Sparse Tensor Streams

Source code for SliceNStitch, described in the paper [SliceNStitch: Continuous CP Decomposition of Sparse Tensor Streams]()

## Requirements

- C/C++17 compiler
- CMake
- Git

## Used Libraries

- Eigen (<https://gitlab.com/libeigen/eigen>)
- yaml-cpp (<https://github.com/jbeder/yaml-cpp>)

## Download

```bash
git clone --recursive https://github.com/DMLab-Tensor/SliceNStitch.git
```

## Generate Build System

To generate the build system, run the following command on your terminal:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
```

After that, you can build the source code using the build automation software (e.g, Make, Ninja).

## Execution

To test the algorithms in the paper, run the following command on your terminal:

```bash
./SliceNStitch [config_file]
```

## Config File

### Example config file

```yaml
# test.yaml
data:
    filePath: "nyt_2019.csv"  # DataStream file
    nonTempDim: [265, 265]  # non temporal dimension
    tempDim: 5184000  # Temporal size of data stream

tensor:
    unitNum: 10  # The number of indices in the time mode (W)
    unitSize: 3600  # Period (T)
    rank: 20  # rank of CPD (R)

algorithm:
    name: "SNS_RND+"
    settings:  # Details are described in the next section
        numSample: 20
        clipping: 1000.0

test:
    outputPath: "out.txt"  # Path of the output file
    startTime: 0  # Starting time of the input data to be processed
    numEpoch: 180000  # The number of epochs
    checkPeriod: 1  # Period of printing the error
    updatePeriod: 1  # Period of updating the tensor
    random: true  # Randomness of the initial points
```

### Examples of Possible Algorithms

```yaml
algorithm:
    name: "ALS"
    settings:
```

```yaml
algorithm:
    name: "SNS_MAT"
    settings:
        numIter: 1
```

```yaml
algorithm:
    name: "SNS_VEC"
    settings:
```

```yaml
algorithm:
    name: "SNS_VEC+"
    settings:
        clipping: 1000.0  # Clipping value (η)
```

```yaml
algorithm:
    name: "SNS_RND"
    settings:
        numSample: 20  # Threshold (θ)
```

```yaml
algorithm:
    name: "SNS_RND+"
    settings:
        numSample: 20  # Threshold (θ)
        clipping: 1000.0  # Clipping value (η)
```

## Data Stream File

It should be CSV format file.

- First (N-1) columns should represent the coordinate of input
- Last column should represent the value of input

Since the test data stream files are very large, please download the files via [Dropbox Link]().
