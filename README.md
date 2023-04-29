![CI/CD Pipeline](https://github.com/athletedecoded/rusty-pipe/actions/workflows/az_deploy.yml/badge.svg)

# Rusty Pipe [WiP]

E2E ML Pipeline -- from data to distroless deploy

![image](./assets/rusty-pipe.png)


## Train

**Download example dataset**

```
$ cd train
$ make dataset
```

**Train model**

```
$ cargo run hymenoptera_data
```

**Convert model to .ot for Deploy**

```
$ python3 -m venv ~/.venv
$ source ~/.venv/bin/activate
$ pip install -r requirements.txt
$ make models
$ cd ../deploy
$ zip -r model.zip model.ot
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

*Gotcha: Ensure Repo > Settings > Actions > General > Allow all actions*


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
- [ ] Models: improve CNN, fix VGG, pass model as CLI param
- [ ] Dynamic class generation --> txt file

**Deploy**
- [ ] Fix GHA deploy -- look into GHLFS for model.zip
- [ ] Switch from ot to onnx rt
- [ ] Load testing



## References

* [Rusty Deploy example](https://github.com/nogibjj/rusty-deploy)
* [tch-rs examples](https://github.com/LaurentMazare/tch-rs/tree/main/examples)