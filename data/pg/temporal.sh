#!/bin/bash
#
# Temporal series of the Bitcoin payment graph.
#
# The chunks under ${PIPELINE}/chunks are six-month slices of the transaction
# list, so an even INDEX is a whole number of years. For every even INDEX in
# 2..28 this script builds the cumulative payment graph over chunks 1..INDEX,
# its accessory structures and its transpose, then estimates the average
# distance and the diameter with `unipairs` and the harmonic centralities with
# `harmonic`.
#
# Everything lands in ${PIPELINE}/graph:
#
#   pg_INDEX.{graph,offsets,properties,ef,dcf}      the graph
#   pg_INDEX-t.{graph,offsets,properties,ef,dcf}    its transpose (the -t.dcf
#                                                   is for later work on the
#                                                   transpose; nothing here
#                                                   reads it)
#   pg_nm_INDEX.tsv                                 the node map
#   avgdist-uni-pg_INDEX.out                        the unipairs estimate
#   harmonic-pg-INDEX.out                           one node/centrality pair per line
#
# and the stderr of every stage is teed to ${PIPELINE}/logs/temporal_pg_INDEX.log,
# which is where the |V|, |E| and |S| of each run are recorded.
#
# NOTHING IS REBUILT THAT IS ALREADY THERE. Every artifact is made only if it
# is missing, empty, or older than the file it derives from -- `make`'s rule --
# or if a marker says the run that made it was interrupted.  So the script is
# restartable: kill it and run it again and it resumes.  Three consequences
# worth knowing:
#
#   * pg_20 and pg_28 already exist and are KEPT, so the series will mix them
#     with graphs built now.  Delete them by hand first if you want all
#     fourteen points produced by one version of the tools.
#   * the mtime rule is what keeps a freshly built graph from being read
#     through a previous run's stale .ef.
#   * the PARAMETERS are not part of the rule.  Change EPSILON and nothing
#     re-runs, so the series would silently mix points computed at different
#     sample sizes.  After changing it, delete the results you want redone:
#     `rm graph/avgdist-uni-pg_*.out graph/harmonic-pg-*.out`.
#
# The text edge list pg_el_INDEX.tsv is never written: it reaches 182 GB at
# INDEX=28 and nothing downstream reads it.  The node map is kept, since it is
# what maps node ids back to UTXOs.
#
#   *** THE ONLY FILES THIS SCRIPT DELETES ***
#
#   The stale edge lists pg_el_INDEX.tsv of the indices being processed --
#   today that is pg_el_20.tsv (72 GB) and pg_el_28.tsv (182 GB).  Nothing
#   reads them and the series needs the space.  Comment out the loop marked
#   RECLAIM below to keep them.
#
# The run takes days.  Start it detached and watch the log:
#
#   nohup ./data/pg/temporal.sh > temporal.log 2>&1 &
#   tail -f temporal.log
#
# A failing index is reported and skipped; the script carries on with the next
# one and exits non-zero at the end if any of them failed.  To do only some
# points, pass them as arguments:
#
#   ./data/pg/temporal.sh 22 24 26 28
#

# No `set -e`: a failing index must not kill the rest of the series.  Every
# stage is checked explicitly instead.
set -u
set -o pipefail

PIPELINE=/data/bitcoin/2022/utxo-spllitting-pipeline
GRAPH_DIR="${PIPELINE}/graph"
LOG_DIR="${PIPELINE}/logs"
TMP_DIR="${PIPELINE}/tmp"
UTXO2WEBGRAPH="${HOME:-}/Developer/working-copies/utxo2webgraph-rs/target/release/utxo2webgraph"
WEBGRAPH=/data/bitcoin/mnocentini-home/Developer/working-copies/webgraph-rs/target/release/webgraph

THREADS=112
# `harmonic` holds one f64 per node plus, per running visit, a bit vector and a
# BFS frontier -- and on pg_28-t the frontier is the big one, about 2.7 GB per
# visit, so the thread count is really a memory budget.  8 is what the existing
# benchmarks use.  Measured on the real pg-t graphs, the harmonic stage over
# indices 8..28 costs roughly 60 h at 8 threads, 17 h at 32 (~86 GB peak) and
# 9 h at 112 (which would not fit).  Raise this to 32 if the box is yours.
HARMONIC_THREADS=8
EPSILON=0.1

# Below this many nodes the payment graph is still mostly isolated UTXOs --
# pg_1 has 18545 nodes and 1094 arcs -- and a uniformly drawn pair is almost
# never connected.  `unipairs` then cannot find a connected pair at all and
# says so; `harmonic` is worse, because it silently returns almost nothing: on
# pg_1-t a sampled run reaches 15 nodes out of 18545, an exact one 1033.
#
# So the two tools are handled differently, on purpose.  `unipairs` tries
# sampling and falls back on failure, which is detectable from its exit status.
# `harmonic` cannot be diagnosed that way, so its mode is chosen up front from
# the node count.
#
# The cap is what bounds the cost of that choice.  An exact run does one visit
# per node, and a visit costs about 0.5 us per node it reaches, so on a
# well-connected graph the exact cost grows as |V|^2: measured, that is about
# half an hour at 10^6 nodes and a day and a half at 5*10^6.  A million keeps
# the worst case inside an hour and still covers the two indices that are
# genuinely degenerate (INDEX 2 at ~3.2e4 nodes and INDEX 4 at ~2.7e5).
# INDEX 6, at ~4.9e6 nodes, is sampled -- watch the coverage warning below.
EXACT_MAX_NODES=1000000

# Disk to require before starting, in KiB.  Measured against the real 2.18e9
# node graph: the fourteen graphs, transposes and node maps come to about
# 340 GB (~35 bytes per node), the harmonic outputs to about 80 GB (~30 bytes
# per reached line, a bit over a quarter of the nodes), and the arc sort spills
# another 129 GiB for pg_28 alone, concurrently with all of it.
REQUIRED_SPACE_KIB=$((600 * 1024 * 1024))

log() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    log "$*"
    exit 1
}

# True when ${target} has to be made: a previous attempt was interrupted, or it
# is missing, empty, or older than the file it derives from.
needs_build() {
    local target=$1 source=${2:-}
    [ -e "${target}.incomplete" ] && return 0
    [ -s "${target}" ] || return 0
    [ -n "${source}" ] && [ "${source}" -nt "${target}" ] && return 0
    return 1
}

# Runs a stage that writes straight into ${target}.  `webgraph build ef` and
# `webgraph build dcf` use no temporary file, so an interrupted run leaves a
# truncated artifact that is non-empty and NEWER than its source -- which every
# freshness test would then accept forever, making the index permanently
# unbuildable until someone deletes it by hand.  The marker is written before
# the stage and removed only on success, so an interruption survives it,
# including a SIGKILL, which no trap can catch.
guarded_build() {
    local target=$1; shift
    local rc=0

    : > "${target}.incomplete" || return 1
    "$@" || rc=$?

    if [ "${rc}" -ne 0 ] || [ ! -s "${target}" ]; then
        rm -f "${target}"
        return 1
    fi

    rm -f "${target}.incomplete"
    return 0
}

# True when the BVGraph triple at ${base} is complete and no older than the
# file it derives from.
graph_present() {
    local base=$1 source=${2:-}
    [ -s "${base}.graph" ] && [ -s "${base}.offsets" ] && [ -s "${base}.properties" ] || return 1
    [ -n "${source}" ] && [ "${source}" -nt "${base}.graph" ] && return 1
    return 0
}

run_unipairs() {
    local base=$1 exact=$2
    cargo run --manifest-path "${REPO}/Cargo.toml" --bin unipairs --release -- \
        "${base}.graph" "${THREADS}" "${EPSILON}" "${exact}"
}

run_harmonic() {
    local base=$1 exact=$2 threads=$3
    cargo run --manifest-path "${REPO}/Cargo.toml" --bin harmonic --release -- \
        "${base}.graph" "${threads}" "${EPSILON}" "${exact}"
}

# Reports how much of the graph the centralities actually cover.
#
# The line count is estimated from the file size rather than counted: these
# outputs reach tens of GB and `wc -l` would re-read all of it on every
# restart.  ~30 bytes is the measured mean of a "node<TAB>value" line.
#
# The warning is the only signal that a sampled run was degenerate.  `harmonic`
# exits 0 whatever it finds, so on a graph that is still mostly isolated UTXOs
# it happily writes a handful of lines -- on pg_1-t a sampled run covers 15
# nodes out of 18545 -- and nothing else would say so.
report_coverage() {
    local index=$1 out=$2 nodes=$3 suffix=${4:-}
    local bytes lines

    bytes=$(stat -c %s "${out}" 2> /dev/null) || bytes=0
    lines=$((bytes / 30))

    log "pg_${index}-t: ~${lines} centralities of ${nodes} nodes${suffix:+  (${suffix})}"

    if [ "${nodes}" -gt 0 ] && [ "${lines}" -lt $((nodes / 20)) ]; then
        log "pg_${index}-t: WARNING: fewer than 5% of the nodes were reached; the sample is probably too sparse to be meaningful"
    fi
}

# Builds one point of the series.  Returns non-zero on the first failing stage.
build_index() {
    local index=$1
    local base="${GRAPH_DIR}/pg_${index}"
    local uni_out="${GRAPH_DIR}/avgdist-uni-pg_${index}.out"
    local har_out="${GRAPH_DIR}/harmonic-pg-${index}.out"
    local nodes exact

    # A killed run can leave a partial result behind, and for harmonic that is
    # tens of GB.
    rm -f "${uni_out}.tmp" "${har_out}.tmp"

    if graph_present "${base}"; then
        log "pg_${index}: graph already present, keeping it"
    else
        log "pg_${index}: building from chunks 1..${index}"
        # A partial triple is cleared first, so that a crash leaves obviously
        # missing files rather than a mixture of two runs.  Relative `chunks`
        # and `graph` are this tool's defaults and resolve against the
        # ${PIPELINE} directory we changed into below.
        rm -f "${base}".{graph,offsets,properties}
        "${UTXO2WEBGRAPH}" build "${index}" \
            --threads "${THREADS}" --no-text-edge-list || return 1
    fi

    # `--no-text-edge-list` never writes one, but a previous run of the
    # pipeline may have left one behind, and a stale edge list sitting next to
    # a graph is worse than no edge list at all.
    rm -f "${GRAPH_DIR}/pg_el_${index}.tsv"

    if needs_build "${base}.ef" "${base}.graph"; then
        log "pg_${index}: building ef"
        guarded_build "${base}.ef" \
            "${WEBGRAPH}" build ef "${base}" || return 1
    else
        log "pg_${index}: ef is up to date"
    fi

    if needs_build "${base}.dcf" "${base}.ef"; then
        log "pg_${index}: building dcf"
        guarded_build "${base}.dcf" \
            "${WEBGRAPH}" build dcf -t "${THREADS}" "${base}" || return 1
    else
        log "pg_${index}: dcf is up to date"
    fi

    # --dcf balances the transposition by arcs rather than by nodes, which is
    # what the dcf built just above is for.
    if graph_present "${base}-t" "${base}.graph"; then
        log "pg_${index}-t: transpose is up to date"
    else
        log "pg_${index}: transposing into pg_${index}-t"
        rm -f "${base}"-t.{graph,offsets,properties}
        "${WEBGRAPH}" transform transpose --dcf -t "${THREADS}" \
            "${base}" "${base}-t" || return 1
    fi

    # The transpose comes out with .graph, .offsets and .properties only; it
    # needs its own ef before `harmonic` can load it.
    if needs_build "${base}-t.ef" "${base}-t.graph"; then
        log "pg_${index}-t: building ef"
        guarded_build "${base}-t.ef" \
            "${WEBGRAPH}" build ef "${base}-t" || return 1
    else
        log "pg_${index}-t: ef is up to date"
    fi

    if needs_build "${base}-t.dcf" "${base}-t.ef"; then
        log "pg_${index}-t: building dcf"
        guarded_build "${base}-t.dcf" \
            "${WEBGRAPH}" build dcf -t "${THREADS}" "${base}-t" || return 1
    else
        log "pg_${index}-t: dcf is up to date"
    fi

    nodes=$(sed -n 's/^nodes=//p' "${base}.properties")
    [ -n "${nodes}" ] || { log "pg_${index}: no node count in ${base}.properties"; return 1; }

    # Results are keyed on the .graph they were computed from, not on the .ef:
    # an index is cheap to rebuild and a result is days of compute, so deleting
    # a suspect .ef must not throw the estimate away.  They are written to a
    # temporary file and moved into place only once they hold something, so
    # that a failure never replaces a good result with an empty stub.  `mv`
    # within a directory is atomic.
    if needs_build "${uni_out}" "${base}.graph"; then
        log "pg_${index}: estimating the average distance and the diameter over ${nodes} nodes"
        if ! run_unipairs "${base}" false > "${uni_out}.tmp"; then
            if [ "${nodes}" -le "${EXACT_MAX_NODES}" ]; then
                log "pg_${index}: sampling found no connected pair, falling back to an exact computation"
                run_unipairs "${base}" true > "${uni_out}.tmp" \
                    || { rm -f "${uni_out}.tmp"; return 1; }
            else
                rm -f "${uni_out}.tmp"
                return 1
            fi
        fi
        [ -s "${uni_out}.tmp" ] || { log "pg_${index}: unipairs wrote nothing"; rm -f "${uni_out}.tmp"; return 1; }
        mv -f "${uni_out}.tmp" "${uni_out}" || return 1
        log "pg_${index}: $(cat "${uni_out}")"
    else
        log "pg_${index}: $(cat "${uni_out}")  (already computed)"
    fi

    if needs_build "${har_out}" "${base}-t.graph"; then
        # At this size an exact run is affordable and a sampled one is not
        # meaningful, so it is not a fallback but the chosen mode -- and it can
        # use every core, memory being a non-issue for so few nodes.
        if [ "${nodes}" -le "${EXACT_MAX_NODES}" ]; then
            exact=true
            log "pg_${index}-t: computing the exact harmonic centralities over ${nodes} nodes"
            run_harmonic "${base}-t" "${exact}" "${THREADS}" > "${har_out}.tmp" \
                || { rm -f "${har_out}.tmp"; return 1; }
        else
            exact=false
            log "pg_${index}-t: estimating the harmonic centralities over ${nodes} nodes"
            run_harmonic "${base}-t" "${exact}" "${HARMONIC_THREADS}" > "${har_out}.tmp" \
                || { rm -f "${har_out}.tmp"; return 1; }
        fi
        [ -s "${har_out}.tmp" ] || { log "pg_${index}-t: harmonic wrote nothing"; rm -f "${har_out}.tmp"; return 1; }
        mv -f "${har_out}.tmp" "${har_out}" || return 1
        report_coverage "${index}" "${har_out}" "${nodes}"
    else
        report_coverage "${index}" "${har_out}" "${nodes}" "already computed"
    fi

    return 0
}

# `cargo run` needs this repository's manifest and we are about to `cd` away
# from it, so resolve it now, from the location of this script.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) \
    || die "could not locate the repository from ${BASH_SOURCE[0]}"
[ -f "${REPO}/Cargo.toml" ] || die "${REPO}/Cargo.toml does not exist"

for tool in "${UTXO2WEBGRAPH}" "${WEBGRAPH}"; do
    [ -x "${tool}" ] || die "${tool} is missing or not executable"
done
command -v cargo > /dev/null || die "cargo is not on PATH"
[ -d "${PIPELINE}/chunks" ] || die "${PIPELINE}/chunks does not exist"

if [ $# -gt 0 ]; then
    INDICES=("$@")
else
    INDICES=(2 4 6 8 10 12 14 16 18 20 22 24 26 28)
fi

mkdir -p "${GRAPH_DIR}" "${LOG_DIR}" "${TMP_DIR}" || die "could not create the output directories"

# `webgraph transform transpose` sorts through a bare `tempfile::tempdir()`,
# which honours TMPDIR and otherwise falls back to /tmp -- and /tmp here is a
# 4.9 GB filesystem, far too small for the 8.6e9 arcs of pg_28.  Point it at
# the pipeline's own tmp directory, on the big volume.
export TMPDIR="${TMP_DIR}"

cd "${PIPELINE}" || die "could not cd into ${PIPELINE}"

# One process at a time: two concurrent runs would write the same filenames and
# fight over the same scratch directory.  The lock is released however we exit.
exec 9> "${PIPELINE}/.temporal.lock"
flock -n 9 || die "another temporal.sh is already running"

# Fail before the days of compute rather than after them.
log "checking that unipairs and harmonic build"
cargo build --manifest-path "${REPO}/Cargo.toml" --bin unipairs --bin harmonic --release \
    || die "could not build unipairs and harmonic"

available=$(df --output=avail -k "${GRAPH_DIR}" | tail -1)
if [ "${available}" -lt "${REQUIRED_SPACE_KIB}" ]; then
    log "warning: only $((available / 1024 / 1024)) GiB available on ${GRAPH_DIR}, $((REQUIRED_SPACE_KIB / 1024 / 1024)) GiB recommended"
fi

# RECLAIM.  The edge lists are dead weight now that --no-text-edge-list is
# used, and pg_el_28.tsv alone is 182 GB.  Freeing them up front rather than
# one index at a time keeps the peak down.
for index in "${INDICES[@]}"; do
    rm -f "${GRAPH_DIR}/pg_el_${index}.tsv"
done

failed=()

for INDEX in "${INDICES[@]}"; do
    log "================ INDEX ${INDEX} ================"
    if build_index "${INDEX}" 2> >(tee -a "${LOG_DIR}/temporal_pg_${INDEX}.log" >&2); then
        log "================ INDEX ${INDEX} done ================"
    else
        log "!!!!!!!! INDEX ${INDEX} FAILED, carrying on !!!!!!!!"
        failed+=("${INDEX}")
    fi
done

if [ ${#failed[@]} -eq 0 ]; then
    log "all indices completed: ${INDICES[*]}"
else
    log "failed indices: ${failed[*]}"
    exit 1
fi
