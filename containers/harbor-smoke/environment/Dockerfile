# Mirrors env.def, which is what actually gets built on an Apptainer-only host.
FROM python:3.12-slim-trixie
RUN mkdir -p /app /logs/verifier /logs/agent \
    && chmod 1777 /app /logs /logs/verifier /logs/agent \
    && printf 'placeholder\n' > /app/answer.txt
