# Sandbox for ELT-Bench rollouts: the toolset of ELT-Bench's agents/SWE-agent/docker/elt-swe image
# (whose unpinned python:3.11 base no longer builds), plus dbt-duckdb for local tasks and an
# offline mirror of the Airbyte Terraform provider.
FROM python:3.11-bookworm

RUN wget -qO- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com bookworm main" \
    > /etc/apt/sources.list.d/hashicorp.list \
 && apt-get update && apt-get install -y --no-install-recommends terraform \
 && rm -rf /var/lib/apt/lists/*

# duckdb must match the host version pinned in pyproject.toml.
RUN pip install --no-cache-dir dbt-core dbt-snowflake dbt-duckdb==1.11.1 duckdb==1.5.6 \
    snowflake-connector-python pandas pyyaml requests psycopg2-binary pymongo awscli

# Mirror the Airbyte provider so `terraform init` works offline and fast.
RUN mkdir /tmp/tf && cd /tmp/tf \
 && printf 'terraform {\n  required_providers {\n    airbyte = {\n      source  = "airbytehq/airbyte"\n      version = "0.6.5"\n    }\n  }\n}\n' > main.tf \
 && terraform providers mirror /opt/terraform-mirror && rm -rf /tmp/tf
COPY terraformrc /root/.terraformrc

WORKDIR /workspace
