# Go REST API (cmd/api). Two stages: build a static binary with the Go 1.22 toolchain, then copy
# only that binary into a small runtime image.
#
# Configuration comes only from the runtime environment (API_ADDR, RAG_SERVICE_URL, POSTGRES_*):
# nothing secret is a build argument or part of a layer, and .env is excluded from the build
# context by .dockerignore. The API never needs the OpenAI key.
FROM golang:1.22-alpine AS build

# GOTOOLCHAIN=local: build with this image's Go and fail rather than download a newer toolchain.
# CGO_ENABLED=0: a static binary, so the runtime image needs no C library from the build stage.
ENV GOTOOLCHAIN=local \
    CGO_ENABLED=0 \
    GOOS=linux

WORKDIR /src

# Modules first, so source changes do not re-download dependencies.
COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ cmd/
COPY internal/ internal/
RUN go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api

FROM alpine:3.20

# ca-certificates for any HTTPS the binary makes; busybox wget (part of the base) serves the
# Compose healthcheck. The service runs as an unprivileged user.
RUN apk add --no-cache ca-certificates \
    && adduser -D -H -u 10002 -s /sbin/nologin api

COPY --from=build /out/api /usr/local/bin/api

USER api
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/api"]
