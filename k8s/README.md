# Kubernetes Examples

In [the PEFT/StarCoder example](../../src/hf_libraries_demo/experiments/peft), I (try to) optimize performance for use 
with a single A100 GPU. For me to access an A100, I have to do so via the 
[Nautilus cluster](https://portal.nrp-nautilus.io/).

Here is my kubernetes & docker setups for doing this, some parts may be useful for your own or not.

## Bare minimum k8s job

Since some examples use containers, I assembled an extremely simple job that can be run with
this repo packaged as a Docker container in [`demo_job.yml`](./demo_job.yml). It executes a function defined in
[the packaging demo](../src/hf_libraries_demo/package_demo) to print a formatted string.

1. [Dockerfile](./Dockerfile) defines the image.
2. [Github Action](../.github/workflows/docker_build.yml) builds the image and pushes to DockerHub/Nautilus Gitlab 
container registry.
3. [Demo Job](demo_job.yml) defines a tiny kubernetes job which uses the image, including the command to execute.

Execute the job via `kubectl create -f k8s/demo_job.yml`. This may require adaptation for your
k8s environment.

## Using an A100 GPU

In [peft/starcoder_base_example.yml](./peft/starcoder_base_example.yml), I create a job which
can be used to run the base peft example I created (with TFLOPs calculation) at 
[src/hf_libraries_demo/experiments/peft/base_with_tflops.py](../src/hf_libraries_demo/experiments/peft/base_with_tflops.py).
This includes a few new additions:
- [Specifying an affinity for nodes with A100s](./peft/starcoder_base_example.yml#L14)
- [Pulling the W&B API Key from a secret](./peft/starcoder_base_example.yml#27)
  - Creation of secret not shown, but some [more info here](https://kubernetes.io/docs/concepts/configuration/secret/). I use [Lens](https://k8slens.dev/) to make some of this easier, but it is not particularly light-weight
- Increasing our CPU/Memory requests and specifying we need 1 GPU (affinity handles type)
- Adjusting command executed to log in to huggingface and set its cache directories to a path on a mounted volume. This allows re-use
of downloaded weights and datasets on subsequent job runs.
- mounting the volume mentioned above (`volumeMounts`)
- [A toleration](./peft/starcoder_base_example.yml#27) which prevents the job from running if no A100s are available yet