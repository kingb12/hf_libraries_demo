# Using an A100 on Nautilus with Kubernetes

In [the PEFT/StarCoder example](../../src/hf_libraries_demo/experiments/peft), I (try to) optimize performance for use 
with a single A100 GPU. For me to access an A100, I have to do so via the 
[Nautilus cluster](https://portal.nrp-nautilus.io/).

Here is my kubernetes & docker setups for doing this, some parts may be useful for your own or not.

1. 
2. Docker Build Github Action: automatically creates and pushes a 