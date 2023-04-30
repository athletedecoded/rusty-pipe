# image:
# 	docker build -t rusty-pipe .

# docker:
# 	docker run -p 8080:8080 rusty-pipe

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