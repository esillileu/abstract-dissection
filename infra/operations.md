# External operator handoff and recovery requirements

These are requirements for the external service operator, not procedures owned or
executed by this repository. The repository consumes resulting endpoints and
maintains only study-domain schema migrations.
Current topology is evidenced by the separately managed deployment at c49c621.
The coordinated backup policy below is a new operational requirement: the inspected
deployment did not establish an existing scheduled backup or tested recovery SLA.

## Bootstrap and deployment

1. The external operator allocates persistent PostgreSQL, master, volume and filer directories/volumes
   under the service account. Record filesystem ownership and image digests.
   Provision a rootless Podman network and a lingering systemd user manager.
2. The external operator creates secrets out of band, with restrictive permissions. Prepare Quadlets
   with the dependencies and loopback bindings in README.md. Install them under
   the service account's containers/systemd configuration, then have the operator
   reload its user systemd manager. Review generated units before starting them.
3. The external operator starts PostgreSQL and verifies pg_isready. It provisions the three database roles and
   databases with isolated grants. Start master, volume, filer, then S3. Verify
   persisted mount destinations before putting objects into the stack.
4. The external operator uses the bootstrap identity to create the MLflow and corpus buckets. Verify each
   application can PUT, GET, LIST its own sentinel and cannot access other buckets;
   verify anonymous access fails. Download each sentinel and compare SHA-256.
5. The external operator starts F1/F2 MLflow with dedicated backend URLs, pgpass secrets and artifact
   destinations. Check /mlflow-f1/health on port 5001 and /mlflow-f2/health on 5002.
   Bootstrap F2 catalog then corpus through the study CLI. Record migration ledgers.
6. The external operator runs one sentinel MLflow run per service, logs metrics and an artifact, downloads
   the artifact and compare SHA-256. Keep the sentinel runs as deployment evidence.
   Check Tailnet/tunnel paths from a worker as well as local endpoints. Record the
   deployment revision, image digests, health and access-control outcomes.

The inspected health probes are master /cluster/status, volume /status, filer
/healthz and S3 /healthz. Check user-systemd unit status and logs, container health,
loopback-only listeners, disk space and read/write errors. Endpoint health alone
does not prove artifact durability: upload/download checks are required.

## Incidents and upgrades

For a failing request, identify its layer: tunnel/TLS, MLflow API, database, S3
signature/permission, filer or volume. Compare configured endpoint host, clock,
region and path addressing for SigV4 errors. Check secret names and grants without
printing secret values. For missing objects, compare DB URI and SHA-256 with S3
bytes and inspect filer/volume health. Preserve manifests and logs for diagnosis.
Stop writers on storage exhaustion or inconsistent lineage; do not mark runs durable.

Before upgrades, take a coordinated backup and record old configuration and image
digests. Test migrations and sentinel roundtrips in an isolated restored environment.
Only the operator reloads/restarts live units. If new writes occurred after a storage
change, reverting configuration alone is unsafe: recover a consistent database and
object set or copy missing bytes and verify hashes first. Never use blind sync/delete
to reconcile a partially failed storage transition.

## Coordinated backup and restore

1. Define recovery point/time objectives, retention and backup frequency with the
   operator. Schedule encrypted off-host backups; alert on failures and overdue
   restore drills. These must be implemented operationally, not assumed present.
2. Quiesce corpus and MLflow writers and wait for in-flight uploads/transactions.
   Record UTC cutoff, service/image revisions, PostgreSQL versions, run IDs and
   corpus lineage heads. Export object URI/byte-size/SHA-256 inventories and
   migration ledger versions; do not use S3 ETag as a content checksum.
3. With writers still stopped, take pg_dump custom-format backups of mlflow_f1,
   mlflow_f2 and f2 plus a protected role/grant export. Shut down SeaweedFS services
   cleanly and snapshot/copy master, filer and every volume directory together.
   Preserve immutable object backups plus key inventories if using logical S3
   backups instead. Record the mapping from the DB snapshot to its object snapshot.
4. Check backup hashes, encrypt and copy off-host with a retention manifest. Store
   secrets separately in the credential backup system. Resume services in dependency
   order, then writers after health checks. A single-host snapshot alone is not enough.
5. Restore into an isolated network and empty volumes. Restore roles/grants and all
   three databases with compatible PostgreSQL tooling; restore all SeaweedFS state
   from the same cutoff before starting master→volume→filer→S3. Alternatively rebuild
   object storage from the logical backup with identical bucket/key mappings.
   Inject restored/rotated credentials and start MLflow against these restored services.
6. Verify catalog/corpus migration ledgers without editing SQL, FK integrity and
   lineage references, immutable validation profiles, canonical resource versions,
   artifact counts and sizes. Download referenced corpus and MLflow artifacts and
   compare their recorded SHA-256 hashes. Check missing/orphan objects explicitly;
   never silently discard them. Validate a representative complete run's checkpoint
   manifests and pointer targets, then run sentinel upload/download tests.
7. Document the achieved recovery point/time and every discrepancy. Cut over only
   after the database and object inventories agree and the operator approves the
   recovered state. Retain the previous deployment and backup until validation ends.
