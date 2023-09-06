# Workflows

## [Docker Build](./workflows/docker_build.yml)

For some experiments demonstrated, I need resources only available to me via the 
[Nautilus Cluster](https://portal.nrp-nautilus.io/). [`docker_build.yml`](./workflows/docker_build.yml) 
builds my docker image for using this repo on the cluster w/ Kubernetes (see [../k8s](../k8s))

Requires setting up the following Github Secrets:

- `DOCKERHUB_TOKEN`: an access token for Dockerhub, can be created [here](https://hub.docker.com/settings/security?generateToken=true)
- `GITLAB_DOCKER_REGISTRY_TOKEN`: a project access token for [Nautilus Gitlab](https://gitlab.nrp-nautilus.io/). I typically do the following:
  - Create a project in Gitlab to hold my containers ([example](https://gitlab.nrp-nautilus.io/kingb12/hf_libraries_demo))
  - Add a project access token (`Settings/Access Tokens` from repo page, I give `read_registry` and `write_registry` permissions)
