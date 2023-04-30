FROM rust:latest as builder

WORKDIR /usr/src/app
COPY deploy/. .

# Install Libtorch (PyTorch C++ backend) v2.0.0
RUN curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip && \
    mv libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Get model weights
RUN wget https://rustypipe.blob.core.windows.net/models/model.zip && \
    unzip model.zip && \
    rm model.zip

RUN cargo build --release

FROM gcr.io/distroless/cc-debian11
COPY --from=builder /usr/src/app/target/release/deploy /usr/local/bin/deploy
COPY --from=builder /usr/src/app/libtorch /usr/local/libtorch
COPY --from=builder /usr/src/app/model.ot /model.ot

# Set LIBTORCH environment variable
ENV LD_LIBRARY_PATH /usr/local/libtorch/lib/

#export this actix web service to port 8080 and 0.0.0.0
EXPOSE 8080
CMD ["deploy"]