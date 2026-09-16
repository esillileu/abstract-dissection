# External service handoff specification

This repository records the interface and handoff information for services that
are operated outside this repository. Operators maintain the live deployment
separately. Evidence: `/opt/infra` documentation and Quadlets,
revision c49c621. Commit ce4b444 removed the former local MLflow operations;
no additional motive is inferred. Compose and one-off migration tooling are not restored.
This directory is not the service owner, deployment authority, or source of truth
for live infrastructure. No deployment, restart, data migration or secret rotation
is part of this refactor.

Rootless Podman containers run under the dedicated `svc-infra` account with a
lingering user systemd manager. The account can have a non-login shell; operators
enter an explicit Bash with its correct XDG_RUNTIME_DIR and user bus. Repository
editors prepare configuration; the service operator installs Quadlets and secrets.
UIDs, paths, endpoint names and image pins are deployment inputs, not portable defaults.

| Service | Internal network | Host loopback | Durable state |
| --- | --- | --- | --- |
| PostgreSQL 18 | postgres:5432 | 5432 | PostgreSQL volume |
| F1 MLflow | service:5000 | 5001, prefix /mlflow-f1 | mlflow_f1 DB and mlflow-f1-artifacts bucket |
| F2 MLflow | service:5000 | 5002, prefix /mlflow-f2 | mlflow_f2 DB and mlflow-f2-artifacts bucket |
| SeaweedFS master | seaweed-master:9333 | none | master metadata |
| SeaweedFS volume | seaweed-volume-ssd:8080 | none | volume bytes and indexes |
| SeaweedFS filer | seaweed-filer:8888 | none | filer metadata |
| SeaweedFS S3 | seaweed-s3:8333 | 9000 | gateway config supplied as secret |

All containers share a private Podman network. Master precedes volume; filer
requires master and volume; S3 requires filer; MLflow requires PostgreSQL and S3.
SeaweedFS replication 000 is a single-server configuration, not HA or a backup.
The inspected deployment pins SeaweedFS 4.41 by digest. Operators must record and
validate image digests, PostgreSQL major version and the MLflow image with its
PostgreSQL driver and boto3 before deploying an equivalent stack.

## Database and object permissions

The external database operator may create separate LOGIN roles and databases for
mlflow_f1, mlflow_f2 and f2.
Use SCRAM passwords generated outside Git. Revoke PUBLIC database CONNECT and
schema CREATE; grant each application CONNECT only to its own database and
USAGE/CREATE on its owned schema. MLflow roles own their backend tables and run
MLflow migrations. F2 supplies and maintains only its catalog/corpus schema
definitions and migration ledgers; it does not administer the database instance.
The external operator provisions roles/databases; it does not define F2 domain tables.
Bootstrap the unified F2 database with catalog migrations before corpus migrations:

```bash
uv run repro f2 catalog migrate
uv run repro f2 corpus migrate
```

Run these with F2_DATABASE_URL injected by the secret manager. Re-running is
idempotent; SQL filenames and ledger versions remain study-owned.

Supply PostgreSQL admin password as a mounted Podman secret. MLflow receives
PGPASSFILE pointing at a mode-0600 secret, and separate AWS_ACCESS_KEY_ID and
AWS_SECRET_ACCESS_KEY environment secrets. Its backend URI contains no password.
SeaweedFS receives a read-only IAM JSON secret. Each MLflow identity has
Read/Write/List/Tagging only for its own bucket. The corpus identity has
Read/Write/List only for f2-corpus. Restricted Gigaword credentials must use a
separate authorized bucket and identity, never the public corpus credential.
A bootstrap admin creates buckets; applications do not receive its credential.
Deny anonymous access and cross-bucket access, and verify both before use.

## Connectivity

Bind host ports only to 127.0.0.1. Tailnet TLS proxies may expose the MLflow paths
and S3 at `https://<tailnet-host>:9000`; PostgreSQL uses an explicitly configured
Tailnet TCP forward. Do not bind SeaweedFS internal ports on the host.
MLflow allowed-hosts and CORS origins must match the selected endpoint placeholders.
S3 uses AWS SigV4 path-style addressing. The endpoint used for a presigned URL
must be reachable by the client without changing its signed host/path.

Optional reverse SSH tunnels request loopback listeners on the remote worker:
15001→5001, 15002→5002, 15432→5432, 19000→9000. Use BatchMode,
ExitOnForwardFailure and keepalives; remote GatewayPorts must preserve loopback
bindings. These connections are opt-in, not enabled by installing configuration.
Check listener addresses and verify listeners disappear when a tunnel stops.

The application environment contract is in
[storage guidelines](../docs/guidelines/path-and-storage.md). Real endpoints and
credentials belong in deployment secrets or uncommitted environment files.
See [operations](operations.md) for bootstrap, incidents and recovery.
