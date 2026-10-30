# Gate between sandboxes and the shared local Airbyte: one workspace per rollout (see gate.py).
FROM python:3.12-alpine
COPY gate.py /gate.py
EXPOSE 80
CMD ["python", "-u", "/gate.py"]
