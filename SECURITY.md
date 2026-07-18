# Security Policy

## Supported versions

This project is pre-1.0. Security fixes land on `main`; track the latest commit.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/ahmeddoghri/kvsqueeze/security/advisories/new)
or email the maintainer. Include a description of the issue and its impact,
steps to reproduce (a minimal proof-of-concept helps), and any suggested fix.

You can expect an acknowledgement within a few days. Once a fix is out you will
be credited unless you would rather stay anonymous.

## Scope notes

kvsqueeze is a pure-stdlib benchmark with no runtime dependencies and makes no
network calls. It runs deterministic simulations on integer token streams, so
there is very little attack surface here. The usual care applies if you feed it
data from an untrusted source.
