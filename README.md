# llama.cpp for GitHub Copilot

## Introduction

This repository includes a lightweight proxy service that routes requests between local clients and a llama.cpp-based model runner. The proxy provides a stable HTTP API endpoint, handles request translation, and makes it easier to integrate the model into development environments such as GitHub Copilot.

The proxy is implemented in Go and is designed to:

- accept incoming API requests from clients
- forward supported request formats to the local model runner
- translate between API layers and model engine commands
- manage configuration and runtime options in a simple way

The service supports an API specification compatible with `Ollama` version `0.02.7`, making it suitable for clients expecting that exact API contract.

Use the proxy when you want a consistent interface for model inference and better control over request handling, logging, and configuration.
