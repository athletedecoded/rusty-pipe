<!-- ![CI/CD Pipeline](https://github.com/athletedecoded/hf-micro/actions/workflows/az_deploy.yml/badge.svg) -->

# IDS721 Spring 2023 Final Project - Rusty Pipe

E2E ML Pipeline -- from data to distroless deploy

<!-- ![image](./assets/hf-micro.png) -->


## Train

**Download example dataset**

```
$ cd train
$ make dataset
```

**Train model**

TODO: Add model CLI param
```
$ cargo run --m <model> --d <data_dir>
```

**Convert model to Onnx for Deploy**

```
$ python3 -m venv ~/.venv
$ source ~/.venv/bin/activate
$ pip install -r requirements.txt
$ make models
```

## Deploy

**Test Locally**

```
$ cd ../deploy
$ make run
```

**Deploy to Azure Distroless Container**

1. Provision Container App `rusty-pipe` in Azure. Update Container App > Settings > Ingress > Port = 8080.

2. From Azure CLI, generate Service Principle Credentials. Capture output and add to Github Repo as `AZURE_CREDENTIALS` Actions Repository Secret.
```
az ad sp create-for-rbac --name "RustyPipeAccess" --role contributor --scopes /subscriptions/$AZURE_SUBSCRIPTION_ID --sdk-auth
```

3. Generate GitHub [PAT with write:packages access](https://github.com/settings/tokens/new?description=RustyPipe+Azure+access&scopes=write:packages). Capture output and add to Github Repo as `PAT` Actions Repository Secret.

4. Push then manually trigger from GitHub Actions > Trigger Auto Deploy from branch: deploy-distro

```
git checkout -b deploy-distro
git push origin deploy-distro
``` 

* Ensure Repo > Settings > Actions > General > Allow all actions


## Useage & Endpoints

Supported endpoints to base URL https://localhost:8080

**GET /** -- Homepage

**POST /predict** -- Predict Image

```
curl -X POST -H "Content-Type: multipart/form-data" -F "image=@deploy/ant.jpg" http://127.0.0.1:8080/predict
```


## ToDos

**Train**
- [ ] Dataset: create tch dataloader that takes train_val split with class subdirectories
- [ ] Models: improve CNN, add VGG, pass model as CLI param
- [ ] Write labels to labels.txt --> cp to deploy

**Deploy**
- [ ] Switch from ot to onnx rt



## References

* [Rusty Deploy example](https://github.com/nogibjj/rusty-deploy)
* [tch-rs examples](https://github.com/LaurentMazare/tch-rs/tree/main/examples)