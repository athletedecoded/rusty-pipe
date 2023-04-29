models:
	python3 convert.py
	cp model.onnx model.ot ../deploy

dataset:
	wget https://download.pytorch.org/tutorial/hymenoptera_data.zip &&\
	unzip hymenoptera_data.zip &&\
	rm hymenoptera_data.zip

format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

test:
	cargo test --quiet

run:
	cargo run 

release:
	cargo build --release

all: format lint test run