FROM rust:latest as builder
ARG token
ENV HF_ACCESS_TOKEN $token
WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# Use distroless debian11 whih includes ca-certificates
FROM gcr.io/distroless/cc-debian11
COPY --from=builder /usr/src/app/target/release/hf-micro /usr/local/bin/hf-micro

#export this actix web service to port 8080 and 0.0.0.0
EXPOSE 8080
CMD ["hf-micro"]