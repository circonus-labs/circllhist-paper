# Circllhist-Paper

A pdf version of the paper is available [here](./circllhist.pdf).

## Evaluation

Datasets and source code used for the evaluation are provided in this repository.
To reproduce the evaluation results, you can use the following docker container:

```
docker run -it --rm -p 9999:9999 -p 9998:9998 -v  "$(pwd):/home/jovyan/work" heinrichhartmann/circllhisteval
```

The Dockerfile used to create the container image is available under ./evaluation/Dockerfile.
The precise image used for the evaluation on our machines was uploaded to dockerhub under the above tag.
