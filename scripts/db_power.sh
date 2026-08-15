#!/usr/bin/env bash
# Start/stop the Cloud SQL instance to control cost between demos.
#
# Cloud SQL is ~85% of this project's GCP spend (~$9.40/mo of ~$10) because,
# unlike Cloud Run (minScale=0, ~free when idle), a database instance bills
# for wall-clock time whether or not anything queries it. Stopping it saves
# the instance charge but NOT the disk: the 10GB PD_SSD keeps billing
# (~$1.70/mo) since the data has to persist to be there when you restart.
#
# Usage:
#   ./scripts/db_power.sh status
#   ./scripts/db_power.sh stop
#   ./scripts/db_power.sh start
#
# NOTE: stopping takes BOTH environments down -- prod and staging share this
# one instance (separate databases within it, ragdb / ragdb_staging). While
# stopped, /ready returns 503 and /ask fails on both services; /health still
# returns 200 because it is a pure liveness probe with no dependency checks.
set -euo pipefail

INSTANCE="${CLOUDSQL_INSTANCE:-rag-capstone-db}"
PROJECT="${GCP_PROJECT_ID:-hybrid-rag-505311}"

usage() { echo "usage: $0 {status|stop|start}" >&2; exit 1; }
[ $# -eq 1 ] || usage

state() {
  gcloud sql instances describe "$INSTANCE" --project="$PROJECT" \
    --format="value(state,settings.activationPolicy)"
}

case "$1" in
  status)
    echo "instance: $INSTANCE (project $PROJECT)"
    echo "state / activation policy: $(state)"
    ;;
  stop)
    echo "stopping $INSTANCE ..."
    gcloud sql instances patch "$INSTANCE" --project="$PROJECT" \
      --activation-policy=NEVER --quiet
    echo "stopped. Both prod and staging will now fail /ready until you start it again."
    ;;
  start)
    echo "starting $INSTANCE ..."
    gcloud sql instances patch "$INSTANCE" --project="$PROJECT" \
      --activation-policy=ALWAYS --quiet
    echo "started. Allow ~1-2 min before connections succeed."
    # Cloud Run scales to zero, so the next request generally starts a fresh
    # container with a fresh connection pool. A container still warm from
    # before the stop can hold dead pooled connections -- if /ready keeps
    # failing after the instance is RUNNABLE, force new revisions:
    #   gcloud run services update rag-capstone --region=us-central1 \
    #     --update-env-vars=DB_RESTARTED_AT=$(date +%s)
    ;;
  *) usage ;;
esac
