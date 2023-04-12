backbone:
	wget https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot

dataset:
	wget https://download.pytorch.org/tutorial/hymenoptera_data.zip &&\
	unzip hymenoptera_data.zip &&\
	rm hymenoptera_data.zip

install:
	cargo clean &&\
	cargo build -j 1

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