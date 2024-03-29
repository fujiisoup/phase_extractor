# A simple script for Quantitative Phase Imaging

This contains a simple script to extract the local phase values from the image with interference pattern.

## Example

**original / extracted phase image**

<img src="data/step.bmp" alt="original" width="150">
<img src="data/step_phase.bmp" alt="phase" width="150">

## Requirements

This script requires

+ numpy
+ scipy

**opencv** is necessary for read/write movie files.

In anaconda environment, do
```
conda install -c conda-forge opencv
```

**skimage** is necessary to unwrap the phase from the file.


## Usage

```
python extract_phase.py [path/to/image]
```

This saves the extracted value as an image file with suffix `_amp` (for amplitude) and `_phase` (for phase).

### Phase difference from an reference image

```
python extract_phase.py [path/to/reference_image] [path/to/target_image1] [path/to/target_image2] ...
```

The wild cards can be used to scan the files. For example,
```
python extract_phase.py [path/to/reference_image] path/to/target[0-9][0-9].bmp ...
```
converts the files `path/to/target01.bmp`, `path/to/target02.bmp`, ..., `path/to/target99.bmp`.

The video can be also used for the phase retrieval.
```
python extract_phase.py [path/to/video_file]
```

In this case, the first frame will be used for the reference image.

## Detailed tuning

See `settings.ini`.

### Typical tuning

- `n_waves` number of waves to be averaged. Larger this value, lower the low-path cut-off frequency.
- `format` output type. If `bmp`results will be saved as a `bmp` file with squeezing the result into 8-bit. If `csv`, the result will be a `csv` file.
