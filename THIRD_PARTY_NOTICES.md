# Third-party notices

The root [MIT license](LICENSE) covers this project's code and original interface artwork.
It does not replace the licenses of datasets, bundled artifacts, or installed dependencies.

## MaleCNS v1.0

Source: [Male CNS Connectome Project](https://male-cns.janelia.org/) and
[downloads](https://male-cns.janelia.org/download/).

Attribution: FlyEM at HHMI Janelia Research Campus, the University of Cambridge Department
of Zoology, the MRC Laboratory of Molecular Biology, Google Research, and collaborators.
The project lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`artifacts/demo/auditory.npz` is a derived selection of that dataset: 30,000 neurons with
1,348,769 signed connections. Selection, edge filtering, sign assignment, and weight scaling
are project modifications, described in [MODEL_CARD.md](MODEL_CARD.md). This artifact remains
under CC BY 4.0. The original tables are downloaded separately and are not included here.

## Free Spoken Digit Dataset

Source: [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset), by Zohar Jackson
and contributors; revision `26eb9aaf76e81b692f806f9140c2d2777410d7a1`.
The dataset's README specifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The ten WAV files in `artifacts/demo/recordings/` are unmodified recordings from that revision.
Original filenames preserve the digit, speaker, and take identifiers. The full dataset is
downloaded separately for training. Training adds synthetic noise and padding to copies;
those augmented recordings are not distributed in the demo bundle.

The trained `artifacts/demo/model.npz` is released under CC BY-SA 4.0, with attribution to
Rahil Mavani and the FSDD contributors. The graph retains its separate CC BY 4.0 terms.
The benchmark report and artifact manifest are also released under CC BY-SA 4.0.

## Scientific reference and inspiration

The simplified neuron dynamics use parameters from Shiu, Sterne, Spiller et al.,
[*A Drosophila computational brain model reveals sensorimotor processing*](https://doi.org/10.1038/s41586-024-07763-9),
Nature 634, 210–219 (2024). This project uses a different selected connectome and an engineered
audio input; the paper did not validate this spoken-digit experiment.

[Jerry Liu's fly_ocr](https://github.com/jerryjliu/fly_ocr) inspired the connectome-decoder
project and its explanatory presentation. The fly illustration and interface in this repository
were created for The Fly Hears.

## Dependencies

Python dependencies are installed from the versions and source URLs recorded in `uv.lock`.
Their own licenses continue to apply; dependency source code is not vendored here.
Speech detection uses [webrtcvad-wheels](https://github.com/daanzu/py-webrtcvad-wheels), which
includes its Python-wrapper and WebRTC license notices in the installed distribution.

The credited researchers, institutions, speakers, and projects do not endorse this demo.
