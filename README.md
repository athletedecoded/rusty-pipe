![CI/CD Pipeline](https://github.com/athletedecoded/hf-micro/actions/workflows/az_deploy.yml/badge.svg)

# IDS721 Spring 2023 Final Project - Rusty Pipe

E2E ML Pipe -- from data to distroless deploy

<!-- ![image](./assets/hf-micro.png) -->


## Setup

**Install**
```
$ make install
```

**Convert .pt to .ot**
```
$ python3 -m venv ~/.venv
$ source ~/.venv/bin/activate
```


## Useage & Endpoints

Supported endpoints to base URL https://localhost:8080

**GET /** -- Homepage


## ToDos

- [x] Distroless CI/CD skeleton



## References

* [Actix extractors](https://actix.rs/docs/extractors/)
* [reqwest crate docs](https://crates.io/crates/reqwest)
