# Stable Diffusion Fine Tuning
Approach as a scalable way of automated model training given a dataset and model confiuration. 

Cloud storage has not been implmeneted due to time constraints, but it has been emulated by `app/data_mount/` which gets mounted into the`tmp` directory in the docker container.

Given a dataset (compressed) and config from storage, a request can be made to `/api/v1/fine_tune_stable_diffusion` which will add a task to the queue to train the model and save it to storage once created. There is a script in `app/scripts/` called ` call_fine_tune.py` which can be used to test the endpoint.

This approach is highly scalable because any n number of fine train consumers can be started to work off any number of model training requests that could be made from a web client after uploading the dataset and specifying any non-default configuration for example.

To run, install docker and run the below in the root directory:
```
docker-compose up
```

## Contribution
This project uses pre commits for code formatting, linting and type checking. Please install pre-commits from pypi and run 

```
pre-commit install
```