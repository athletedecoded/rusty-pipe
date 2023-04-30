models:
	python3 convert.py
	zip -r model.zip model.ot

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