FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY sloforge-gateway /usr/local/bin/sloforge-gateway
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/sloforge-gateway"]
