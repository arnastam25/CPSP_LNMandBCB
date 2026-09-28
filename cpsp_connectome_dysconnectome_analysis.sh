#!/usr/bin/env bash
set -Eeuo pipefail

# CPSP connectome and dysconnectome analysis
# Final LNM, BCB, localisation, robustness and spatial analyses.
# Usage: bash cpsp_connectome_dysconnectome_analysis.sh [primary|localisation|robustness|spatial]

MODE="${1:-primary}"
WORK="${CPSP_CONNECTOME_WORK:-${TMPDIR:-/tmp}/cpsp_connectome_${USER:-user}}"
mkdir -p "$WORK"

run_script() {
    local script="$1"
    shift
    case "$script" in
        *.R)  Rscript --vanilla "$WORK/$script" "$@" ;;
        *.py) python3 "$WORK/$script" "$@" ;;
        *)    bash "$WORK/$script" "$@" ;;
    esac
}

submit_script() {
    local script="$1"
    shift
    if command -v sbatch >/dev/null 2>&1; then
        sbatch --wait "$WORK/$script" "$@"
    else
        bash "$WORK/$script" "$@"
    fi
}

cat > "$WORK/build_heat_only_all_branch_main_dataset.py" <<'SOURCE_BUILD_HEAT_ONLY_ALL_BRANCH_MAIN_DATASET_PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
from pathlib import Path

EXPECTED_N = 63
Z_TOL = 1e-8

def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")

def detect_delimiter(path: Path) -> str:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not lines:
        fail(f"empty table: {path}")
    return "\t" if "\t" in lines[0] else ","

def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"table missing or empty: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=detect_delimiter(path))
        fields = list(reader.fieldnames or [])
        rows = [{field: (row.get(field) or "") for field in fields} for row in reader]
    if not fields or not rows:
        fail(f"table has no fields or data rows: {path}")
    return fields, rows

def write_table(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

def canonical_pid(value: object) -> str:
    digits = re.findall(r"\d+", str(value).strip())
    if not digits:
        fail(f"could not canonicalise participant ID: {value!r}")
    return f"sub-{int(digits[-1]):02d}"

def resolve_field(fields: list[str], aliases: list[str], label: str) -> str:
    lookup = {norm(field): field for field in fields}
    for alias in aliases:
        key = norm(alias)
        if key in lookup:
            return lookup[key]
    fail(f"could not resolve {label}; expected one of: {', '.join(aliases)}")

def finite_float(value: object, label: str) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        fail(f"non-numeric {label}: {value!r}")
    if not math.isfinite(number):
        fail(f"non-finite {label}: {value!r}")
    return number

def as_bool(value: object) -> bool:
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y", "PASS", "GREEN"}

def sample_z(values: list[float]) -> list[float]:
    if len(values) < 2:
        fail("cannot standardise fewer than two values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    if not math.isfinite(sd) or sd <= 0:
        fail("predictor has zero or invalid sample SD")
    return [(value - mean) / sd for value in values]

def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx <= 0 or sy <= 0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

PREDICTORS = [
    {
        "modality": "LNM",
        "branch": "LNM_BranchB",
        "role": "primary",
        "source_raw": "LNM_BA3a_BA3c_primary_BranchB_heatOnly",
        "source_z": "z_LNM_BA3a_BA3c_primary_BranchB_heatOnly",
        "canonical_raw": "LNM_BA3a_BA3c_primary_BranchB",
        "canonical_z": "z_LNM_BA3a_BA3c_primary_BranchB",
        "archive_raw": "archived_union_LNM_BranchB_primary_raw",
        "archive_z": "archived_union_LNM_BranchB_primary_z",
    },
    {
        "modality": "LNM",
        "branch": "LNM_BranchA",
        "role": "sensitivity",
        "source_raw": "LNM_BA3a_BA3c_sensitivity_BranchA_heatOnly",
        "source_z": "z_LNM_BA3a_BA3c_sensitivity_BranchA_heatOnly",
        "canonical_raw": "LNM_BA3a_BA3c_sensitivity_BranchA",
        "canonical_z": "z_LNM_BA3a_BA3c_sensitivity_BranchA",
        "archive_raw": "archived_union_LNM_BranchA_sensitivity_raw",
        "archive_z": "archived_union_LNM_BranchA_sensitivity_z",
    },
    {
        "modality": "BCB",
        "branch": "BCB_BranchB",
        "role": "primary",
        "source_raw": "BCB_BA3a_BA3c_primary_BranchB_2mm_heatOnly",
        "source_z": "z_BCB_BA3a_BA3c_primary_BranchB_2mm_heatOnly",
        "canonical_raw": "BCB_BA3a_BA3c_primary_BranchB_2mm",
        "canonical_z": "z_BCB_BA3a_BA3c_primary_BranchB_2mm",
        "archive_raw": "archived_union_BCB_BranchB_primary_2mm_raw",
        "archive_z": "archived_union_BCB_BranchB_primary_2mm_z",
    },
    {
        "modality": "BCB",
        "branch": "BCB_BranchA",
        "role": "sensitivity",
        "source_raw": "BCB_BA3a_BA3c_sensitivity_BranchA_2mm_heatOnly",
        "source_z": "z_BCB_BA3a_BA3c_sensitivity_BranchA_2mm_heatOnly",
        "canonical_raw": "BCB_BA3a_BA3c_sensitivity_BranchA_2mm",
        "canonical_z": "z_BCB_BA3a_BA3c_sensitivity_BranchA_2mm",
        "archive_raw": "archived_union_BCB_BranchA_sensitivity_2mm_raw",
        "archive_z": "archived_union_BCB_BranchA_sensitivity_2mm_z",
    },
    {
        "modality": "BCB",
        "branch": "BCB_BranchC",
        "role": "sensitivity",
        "source_raw": "BCB_BA3a_BA3c_sensitivity_BranchC_1mm_heatOnly",
        "source_z": "z_BCB_BA3a_BA3c_sensitivity_BranchC_1mm_heatOnly",
        "canonical_raw": "BCB_BA3a_BA3c_sensitivity_BranchC_1mm",
        "canonical_z": "z_BCB_BA3a_BA3c_sensitivity_BranchC_1mm",
        "archive_raw": "archived_union_BCB_BranchC_sensitivity_1mm_raw",
        "archive_z": "archived_union_BCB_BranchC_sensitivity_1mm_z",
    },
]

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace all five task-defined heat-or-vibration branch predictors with audited heat-only values."
    )
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--heat-table", required=True, type=Path)
    parser.add_argument("--heat-qc", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--archived-columns", required=True, type=Path)
    args = parser.parse_args()

    main_fields, main_rows = read_table(args.main)
    heat_fields, heat_rows = read_table(args.heat_table)
    qc_fields, qc_rows = read_table(args.heat_qc)

    qc_modality = resolve_field(qc_fields, ["modality"], "QC modality")
    qc_branch = resolve_field(qc_fields, ["branch"], "QC branch")
    qc_verdict = resolve_field(qc_fields, ["verdict"], "QC verdict")
    qc_repro = resolve_field(qc_fields, ["union_reproduction_pass"], "QC union reproduction")
    qc_nfinite = resolve_field(qc_fields, ["n_finite"], "QC finite count")
    qc_mask = resolve_field(qc_fields, ["mask_exact_match"], "QC mask match")

    expected_qc = {(p["modality"], p["branch"]) for p in PREDICTORS}
    seen_qc: dict[tuple[str, str], dict[str, str]] = {}
    for row in qc_rows:
        key = (str(row[qc_modality]).strip(), str(row[qc_branch]).strip())
        if key not in expected_qc:
            continue
        if key in seen_qc:
            fail(f"duplicate QC row for {key}")
        seen_qc[key] = row
    if set(seen_qc) != expected_qc:
        fail(f"QC rows do not cover all five expected predictors; found={sorted(seen_qc)}")
    for key, row in seen_qc.items():
        if str(row[qc_verdict]).strip().upper() != "GREEN":
            fail(f"QC verdict is not GREEN for {key}: {row[qc_verdict]!r}")
        if not as_bool(row[qc_repro]):
            fail(f"union reproduction did not pass for {key}")
        if int(round(finite_float(row[qc_nfinite], f"n_finite for {key}"))) != EXPECTED_N:
            fail(f"expected {EXPECTED_N} finite values for {key}")
        if key[0] == "BCB" and not as_bool(row[qc_mask]):
            fail(f"BCB mask calibration did not match exactly for {key}")

    main_pid = resolve_field(main_fields, ["participant_id", "subject_id", "subject", "participant"], "main participant ID")
    heat_pid = resolve_field(heat_fields, ["participant_id", "subject_id", "subject", "participant"], "heat participant ID")

    for p in PREDICTORS:
        resolve_field(heat_fields, [p["source_raw"]], p["source_raw"])
        resolve_field(heat_fields, [p["source_z"]], p["source_z"])

    main_by_pid: dict[str, dict[str, str]] = {}
    for row in main_rows:
        pid = canonical_pid(row[main_pid])
        if pid in main_by_pid:
            fail(f"duplicate participant in main dataset: {pid}")
        row[main_pid] = pid
        main_by_pid[pid] = row

    heat_by_pid: dict[str, dict[str, str]] = {}
    for row in heat_rows:
        pid = canonical_pid(row[heat_pid])
        if pid in heat_by_pid:
            fail(f"duplicate participant in heat table: {pid}")
        heat_by_pid[pid] = row

    if len(main_by_pid) != EXPECTED_N or len(heat_by_pid) != EXPECTED_N:
        fail(f"expected {EXPECTED_N} participants in both tables; main={len(main_by_pid)}, heat={len(heat_by_pid)}")
    if set(main_by_pid) != set(heat_by_pid):
        fail(
            "participant sets differ; "
            f"missing_heat={sorted(set(main_by_pid) - set(heat_by_pid))}; "
            f"missing_main={sorted(set(heat_by_pid) - set(main_by_pid))}"
        )

    ordered_pids = [canonical_pid(row[main_pid]) for row in main_rows]
    audit_rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []

    for p in PREDICTORS:
        raw_values = [finite_float(heat_by_pid[pid][p["source_raw"]], f"{p['source_raw']} for {pid}") for pid in ordered_pids]
        supplied_z = [finite_float(heat_by_pid[pid][p["source_z"]], f"{p['source_z']} for {pid}") for pid in ordered_pids]
        recomputed_z = sample_z(raw_values)
        max_z_diff = max(abs(a - b) for a, b in zip(supplied_z, recomputed_z))
        if max_z_diff > Z_TOL:
            fail(f"supplied z-scores do not match raw values for {p['branch']}; max difference={max_z_diff}")

        if p["canonical_raw"] not in main_fields:
            fail(f"historical union raw predictor missing from main dataset: {p['canonical_raw']}")
        historical_raw = [finite_float(row.get(p["canonical_raw"]), f"historical {p['canonical_raw']} for {pid}") for row, pid in zip(main_rows, ordered_pids)]
        if p["canonical_z"] in main_fields:
            historical_z = [finite_float(row.get(p["canonical_z"]), f"historical {p['canonical_z']} for {pid}") for row, pid in zip(main_rows, ordered_pids)]
            archive_z_source = "existing_z_column"
        else:
            historical_z = sample_z(historical_raw)
            archive_z_source = "recomputed_from_historical_raw"

        for field in (p["archive_raw"], p["archive_z"], p["canonical_z"]):
            if field not in main_fields:
                main_fields.append(field)

        for index, row in enumerate(main_rows):
            row[p["archive_raw"]] = f"{historical_raw[index]:.17g}"
            row[p["archive_z"]] = f"{historical_z[index]:.17g}"
            row[p["canonical_raw"]] = f"{raw_values[index]:.17g}"
            row[p["canonical_z"]] = f"{supplied_z[index]:.17g}"

        corr = pearson(historical_raw, raw_values)
        audit_rows.append({
            "modality": p["modality"],
            "branch": p["branch"],
            "role": p["role"],
            "heat_source_raw": p["source_raw"],
            "heat_source_z": p["source_z"],
            "canonical_replaced_raw": p["canonical_raw"],
            "canonical_replaced_z": p["canonical_z"],
            "archived_union_raw": p["archive_raw"],
            "archived_union_z": p["archive_z"],
            "archive_z_source": archive_z_source,
            "n_replaced": EXPECTED_N,
            "heat_raw_unique": len(set(round(v, 14) for v in raw_values)),
            "heat_raw_sd": statistics.stdev(raw_values),
            "heat_vs_union_pearson_r": corr,
            "supplied_vs_recomputed_z_max_abs_diff": max_z_diff,
            "qc_verdict": seen_qc[(p["modality"], p["branch"])][qc_verdict],
            "pass": True,
        })
        archive_rows.extend([
            {"archived_column": p["archive_raw"], "source_column": p["canonical_raw"], "content": "historical heat-or-vibration raw predictor"},
            {"archived_column": p["archive_z"], "source_column": p["canonical_z"], "content": "historical heat-or-vibration z-scored predictor"},
        ])

    metadata = {
        "heat_only_all_branches_replacement": "1",
        "heat_only_primary_replacement": "1",
        "primary_task_roi_definition": "HC19_heatOnly_leftS1PlusBA3cTransition_z2p3",
        "task_roi_replacement_scope": "LNM_BranchB;LNM_BranchA;BCB_BranchB;BCB_BranchA;BCB_BranchC",
        "historical_union_predictors_archived": "1",
    }
    for field in metadata:
        if field not in main_fields:
            main_fields.append(field)
    for row in main_rows:
        row.update(metadata)

    write_table(args.out, main_fields, main_rows)
    write_table(args.audit, list(audit_rows[0].keys()), audit_rows)
    write_table(args.archived_columns, ["archived_column", "source_column", "content"], archive_rows)

    provenance = args.audit.with_name("HEAT_ONLY_ALL_BRANCHES_INPUT_PROVENANCE.tsv")
    write_table(provenance, ["input", "path", "sha256"], [
        {"input": "original_main_dataset", "path": str(args.main.resolve()), "sha256": sha256(args.main)},
        {"input": "heat_only_all_branch_table", "path": str(args.heat_table.resolve()), "sha256": sha256(args.heat_table)},
        {"input": "heat_only_all_branch_qc", "path": str(args.heat_qc.resolve()), "sha256": sha256(args.heat_qc)},
        {"input": "prepared_heat_only_main_dataset", "path": str(args.out.resolve()), "sha256": sha256(args.out)},
    ])

    print(f"SUCCESS heat-only all-branch dataset written: {args.out}")
    print(f"participants={len(main_rows)}")
    print(f"predictors_replaced={len(PREDICTORS)}")
    print(f"replacement_audit={args.audit}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
SOURCE_BUILD_HEAT_ONLY_ALL_BRANCH_MAIN_DATASET_PY

cat > "$WORK/cpsp_four_outcome_pc2_prepare.R" <<'SOURCE_CPSP_FOUR_OUTCOME_PC2_PREPARE_R'
#!/usr/bin/env Rscript
options(stringsAsFactors = FALSE, width = 220, scipen = 8)
set.seed(20260724)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) {
  stop(
    "Usage: cpsp_four_outcome_reanalysis.R <main_or_reviewer_dataset.tsv> <s1_dataset.tsv|NONE> <morel_dataset.tsv|NONE> <outdir>",
    call. = FALSE
  )
}

MAIN_DATA <- normalizePath(args[[1]], mustWork = TRUE)
S1_DATA <- if (toupper(args[[2]]) == "NONE") NA_character_ else normalizePath(args[[2]], mustWork = TRUE)
MOREL_DATA <- if (toupper(args[[3]]) == "NONE") NA_character_ else normalizePath(args[[3]], mustWork = TRUE)
OUT <- normalizePath(args[[4]], mustWork = FALSE)

BOOT_REPS <- as.integer(Sys.getenv("BOOTSTRAP_REPS", "2000"))
RUN_BOOTSTRAP <- identical(Sys.getenv("RUN_BOOTSTRAP", "1"), "1")
RUN_FULL_LOO <- identical(Sys.getenv("RUN_FULL_LOO", "1"), "1")

N_CORES <- suppressWarnings(as.integer(Sys.getenv(
  "N_CORES",
  Sys.getenv("SLURM_CPUS_PER_TASK", "1")
)))
if (!is.finite(N_CORES) || N_CORES < 1L) N_CORES <- 1L
N_CORES <- min(N_CORES, 64L)
parallel_lapply <- function(X, FUN, ..., cores = N_CORES) {
  cores <- max(1L, min(as.integer(cores), length(X)))
  if (length(X) <= 1L || cores <= 1L || .Platform$OS.type != "unix") {
    return(lapply(X, FUN, ...))
  }
  parallel::mclapply(
    X,
    FUN,
    ...,
    mc.cores = cores,
    mc.preschedule = TRUE,
    mc.set.seed = TRUE,
    mc.cleanup = TRUE
  )
}

for (d in c("tables", "audit", "plots", "report")) {
  dir.create(file.path(OUT, d), recursive = TRUE, showWarnings = FALSE)
}

log_path <- file.path(OUT, "analysis_console.log")
sink(log_path, type = "output", split = TRUE)
cat("Analysis console output log: ", log_path, "\n", sep = "")

read_any <- function(path) {
  first <- readLines(path, n = 1L, warn = FALSE)
  sep <- if (grepl("\t", first, fixed = TRUE)) "\t" else ","
  read.table(
    path,
    header = TRUE,
    sep = sep,
    quote = "\"",
    comment.char = "",
    check.names = FALSE,
    fill = TRUE,
    na.strings = c("", "NA", "NaN", "n/a", "N/A", "null", ".", "#REF!", "#VALUE!", "#N/A")
  )
}

write_tsv <- function(x, filename, subdir = "tables") {
  path <- file.path(OUT, subdir, filename)
  y <- x
  if (!is.data.frame(y) || ncol(y) == 0L) {
    writeLines("", path)
    return(path)
  }
  list_cols <- names(y)[vapply(y, is.list, logical(1))]
  for (nm in list_cols) {
    y[[nm]] <- vapply(y[[nm]], function(z) paste(as.character(unlist(z)), collapse = "+"), character(1))
  }
  write.table(y, path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
  path
}

num <- function(x) suppressWarnings(as.numeric(as.character(x)))
norm_name <- function(x) gsub("[^a-z0-9]", "", tolower(trimws(as.character(x))))
canon_pid <- function(x) {
  z <- suppressWarnings(as.integer(gsub("[^0-9]", "", as.character(x))))
  ifelse(is.na(z), as.character(x), sprintf("sub-%02d", z))
}

resolve_col <- function(dat, aliases = character(), regex = NULL, required = FALSE, label = "column") {
  lookup <- setNames(names(dat), norm_name(names(dat)))
  for (a in aliases) {
    key <- norm_name(a)
    if (key %in% names(lookup)) return(unname(lookup[[key]]))
  }
  if (!is.null(regex)) {
    hit <- grep(regex, names(dat), ignore.case = TRUE, value = TRUE)
    if (length(hit)) return(hit[[1]])
  }
  if (required) stop("Missing required ", label, ": ", paste(aliases, collapse = " | "), call. = FALSE)
  NA_character_
}

safe_z <- function(x) {
  x <- num(x)
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 0) return(rep(NA_real_, length(x)))
  (x - m) / s
}

safe_inverse <- function(M, tol = sqrt(.Machine$double.eps)) {
  direct <- tryCatch(solve(M), error = function(e) NULL)
  if (!is.null(direct)) return(direct)
  z <- svd(M)
  if (!length(z$d)) return(matrix(NA_real_, nrow(M), ncol(M)))
  cutoff <- max(z$d) * tol
  dinv <- ifelse(z$d > cutoff, 1 / z$d, 0)
  z$v %*% (dinv * t(z$u))
}

bind_rows_base <- function(xs) {
  xs <- xs[!vapply(xs, is.null, logical(1))]
  xs <- xs[vapply(xs, nrow, integer(1)) > 0]
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  filled <- lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (m in miss) x[[m]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, filled)
}

complete_model_data <- function(dat, vars) {
  vars <- unique(vars[!is.na(vars) & nzchar(vars)])
  miss <- setdiff(vars, names(dat))
  if (length(miss)) stop("Missing model variables: ", paste(miss, collapse = ", "), call. = FALSE)
  dat[complete.cases(dat[, vars, drop = FALSE]), vars, drop = FALSE]
}

formula_string <- function(outcome, terms) {
  terms <- terms[!is.na(terms) & nzchar(terms)]
  if (!length(terms)) paste(outcome, "~ 1") else paste(outcome, "~", paste(terms, collapse = " + "))
}

ensure_z <- function(dat, raw_candidates, z_name) {
  if (z_name %in% names(dat) && sum(is.finite(num(dat[[z_name]]))) >= 10) return(dat)
  raw <- resolve_col(dat, aliases = raw_candidates, required = FALSE)
  if (!is.na(raw)) dat[[z_name]] <- safe_z(dat[[raw]])
  dat
}

bh_within <- function(p, group) {
  out <- rep(NA_real_, length(p))
  for (g in unique(group)) {
    idx <- which(group == g & is.finite(p))
    if (length(idx)) out[idx] <- p.adjust(p[idx], method = "BH")
  }
  out
}

holm_within <- function(p, group) {
  out <- rep(NA_real_, length(p))
  for (g in unique(group)) {
    idx <- which(group == g & is.finite(p))
    if (length(idx)) out[idx] <- p.adjust(p[idx], method = "holm")
  }
  out
}

auc_rank <- function(y, p) {
  keep <- is.finite(y) & is.finite(p)
  y <- y[keep]
  p <- p[keep]
  if (length(unique(y)) != 2L) return(NA_real_)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  if (!n1 || !n0) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

hc3_fit <- function(formula, dat, term, model_id = "") {
  fit <- tryCatch(lm(formula, data = dat), error = function(e) NULL)
  if (is.null(fit)) return(data.frame(model_id = model_id, term = term, status = "FAILED_LM"))
  X <- model.matrix(fit)
  e <- residuals(fit)
  h <- hatvalues(fit)
  inv <- safe_inverse(crossprod(X))
  omega <- e^2 / pmax((1 - h)^2, 1e-12)
  vc <- inv %*% crossprod(X, X * omega) %*% inv
  beta <- coef(fit)
  se <- sqrt(pmax(diag(vc), 0))
  df <- max(1, nrow(X) - qr(X)$rank)
  idx <- which(names(beta) == term)
  if (!length(idx)) return(data.frame(model_id = model_id, term = term, status = "TERM_NOT_ESTIMABLE"))
  tval <- beta[idx] / se[idx]
  crit <- qt(0.975, df)
  data.frame(
    model_id = model_id,
    model_type = "OLS_HC3",
    n = nrow(X),
    events = NA,
    parameters = ncol(X),
    term = term,
    estimate = unname(beta[idx]),
    standard_error = unname(se[idx]),
    statistic = unname(tval),
    p_value = 2 * pt(abs(tval), df, lower.tail = FALSE),
    ci_low = unname(beta[idx] - crit * se[idx]),
    ci_high = unname(beta[idx] + crit * se[idx]),
    odds_ratio = NA,
    OR_ci_low = NA,
    OR_ci_high = NA,
    adjusted_R2 = summary(fit)$adj.r.squared,
    AIC = AIC(fit),
    converged = TRUE,
    rank_deficient = qr(X)$rank < ncol(X),
    status = "OK"
  )
}

standard_logistic_fit <- function(formula, dat, term, model_id = "") {
  fit <- tryCatch(glm(formula, data = dat, family = binomial()), error = function(e) NULL)
  if (is.null(fit)) return(data.frame(model_id = model_id, term = term, status = "FAILED_LOGISTIC"))
  tab <- summary(fit)$coefficients
  if (!term %in% rownames(tab)) return(data.frame(model_id = model_id, term = term, status = "TERM_NOT_ESTIMABLE"))
  b <- tab[term, "Estimate"]
  se <- tab[term, "Std. Error"]
  z <- tab[term, "z value"]
  p <- tab[term, "Pr(>|z|)"]
  y <- model.response(model.frame(fit))
  data.frame(
    model_id = model_id,
    model_type = "standard_logistic",
    n = nobs(fit),
    events = sum(y == 1),
    parameters = length(coef(fit)),
    term = term,
    estimate = b,
    standard_error = se,
    statistic = z,
    p_value = p,
    ci_low = b - 1.96 * se,
    ci_high = b + 1.96 * se,
    odds_ratio = exp(b),
    OR_ci_low = exp(b - 1.96 * se),
    OR_ci_high = exp(b + 1.96 * se),
    adjusted_R2 = NA,
    AIC = AIC(fit),
    converged = isTRUE(fit$converged),
    rank_deficient = fit$rank < length(coef(fit)),
    status = ifelse(isTRUE(fit$converged), "OK", "NONCONVERGED")
  )
}

penloglik <- function(beta, X, y) {
  eta <- drop(X %*% beta)
  p <- plogis(pmax(pmin(eta, 35), -35))
  W <- pmax(p * (1 - p), 1e-10)
  I <- crossprod(X, X * W)
  detI <- determinant(I, logarithm = TRUE)
  if (detI$sign <= 0) return(-Inf)
  sum(y * log(pmax(p, 1e-15)) + (1 - y) * log(pmax(1 - p, 1e-15))) + 0.5 * as.numeric(detI$modulus)
}

firth_matrix <- function(X, y, maxit = 300, tol = 1e-8) {
  b <- rep(0, ncol(X))
  conv <- FALSE
  for (it in seq_len(maxit)) {
    eta <- drop(X %*% b)
    p <- plogis(pmax(pmin(eta, 35), -35))
    W <- pmax(p * (1 - p), 1e-10)
    I <- crossprod(X, X * W)
    inv <- safe_inverse(I)
    h <- rowSums((X %*% inv) * X) * W
    score <- crossprod(X, y - p + h * (0.5 - p))
    step <- drop(inv %*% score)
    old <- penloglik(b, X, y)
    fac <- 1
    repeat {
      cand <- b + fac * step
      ll <- penloglik(cand, X, y)
      if (is.finite(ll) && ll >= old - 1e-10) break
      fac <- fac / 2
      if (fac < 1e-8) break
    }
    nb <- b + fac * step
    if (max(abs(nb - b)) < tol) {
      b <- nb
      conv <- TRUE
      break
    }
    b <- nb
  }
  eta <- drop(X %*% b)
  p <- plogis(pmax(pmin(eta, 35), -35))
  W <- pmax(p * (1 - p), 1e-10)
  I <- crossprod(X, X * W)
  vc <- safe_inverse(I)
  list(beta = b, vcov = vc, fitted = p, converged = conv, iterations = it)
}

firth_fit <- function(formula, dat, term, model_id = "") {
  mf <- model.frame(formula, data = dat, na.action = na.omit)
  y <- model.response(mf)
  X <- model.matrix(formula, mf)
  if (length(unique(y)) != 2L) return(data.frame(model_id = model_id, term = term, status = "FAILED_BINARY_OUTCOME"))
  fit <- tryCatch(firth_matrix(X, y), error = function(e) NULL)
  idx <- match(term, colnames(X))
  if (is.null(fit) || is.na(idx)) return(data.frame(model_id = model_id, term = term, status = "TERM_NOT_ESTIMABLE"))
  se <- sqrt(pmax(diag(fit$vcov), 0))
  z <- fit$beta[idx] / se[idx]
  data.frame(
    model_id = model_id,
    model_type = "Firth_Jeffreys",
    n = nrow(X),
    events = sum(y == 1),
    parameters = ncol(X),
    term = term,
    estimate = fit$beta[idx],
    standard_error = se[idx],
    statistic = z,
    p_value = 2 * pnorm(abs(z), lower.tail = FALSE),
    ci_low = fit$beta[idx] - 1.96 * se[idx],
    ci_high = fit$beta[idx] + 1.96 * se[idx],
    odds_ratio = exp(fit$beta[idx]),
    OR_ci_low = exp(fit$beta[idx] - 1.96 * se[idx]),
    OR_ci_high = exp(fit$beta[idx] + 1.96 * se[idx]),
    adjusted_R2 = NA,
    AIC = NA,
    converged = fit$converged,
    rank_deficient = qr(X)$rank < ncol(X),
    status = ifelse(fit$converged, "OK", "NONCONVERGED")
  )
}

huber_fit <- function(formula, dat, term, model_id = "", maxit = 100, tol = 1e-7, k = 1.345) {
  mf <- model.frame(formula, data = dat, na.action = na.omit)
  y <- model.response(mf)
  X <- model.matrix(formula, mf)
  b <- tryCatch(qr.solve(X, y), error = function(e) rep(0, ncol(X)))
  conv <- FALSE
  for (it in seq_len(maxit)) {
    r <- y - drop(X %*% b)
    s <- median(abs(r - median(r))) / 0.6744898
    if (!is.finite(s) || s <= 1e-10) s <- sqrt(mean(r^2))
    u <- r / pmax(s, 1e-10)
    w <- ifelse(abs(u) <= k, 1, k / abs(u))
    Xw <- X * sqrt(w)
    yw <- y * sqrt(w)
    nb <- tryCatch(qr.solve(Xw, yw), error = function(e) b)
    if (max(abs(nb - b)) < tol) {
      b <- nb
      conv <- TRUE
      break
    }
    b <- nb
  }
  idx <- match(term, colnames(X))
  if (is.na(idx)) return(data.frame(model_id = model_id, term = term, status = "TERM_NOT_ESTIMABLE"))
  r <- y - drop(X %*% b)
  df <- max(1, nrow(X) - qr(X)$rank)
  sigma2 <- sum(w * r^2) / df
  vc <- sigma2 * safe_inverse(crossprod(X, X * w))
  se <- sqrt(pmax(diag(vc), 0))
  tval <- b[idx] / se[idx]
  data.frame(
    model_id = model_id,
    model_type = "Huber_IRLS",
    n = nrow(X),
    term = term,
    estimate = b[idx],
    standard_error = se[idx],
    statistic = tval,
    p_value = 2 * pt(abs(tval), df, lower.tail = FALSE),
    ci_low = b[idx] - qt(0.975, df) * se[idx],
    ci_high = b[idx] + qt(0.975, df) * se[idx],
    converged = conv,
    status = ifelse(conv, "OK", "MAXIT")
  )
}

fit_model <- function(dat, outcome, predictor, covars, model_id, binary = FALSE, method = NULL) {
  vars <- unique(c(outcome, predictor, covars))
  d <- tryCatch(complete_model_data(dat, vars), error = function(e) NULL)
  if (is.null(d) || nrow(d) < 15L || length(unique(num(d[[predictor]]))) < 2L) {
    return(data.frame(model_id = model_id, outcome = outcome, predictor = predictor, status = "INSUFFICIENT_DATA"))
  }
  d[[predictor]] <- num(d[[predictor]])
  form <- as.formula(formula_string(outcome, c(predictor, covars)))
  if (binary) {
    out <- if (identical(method, "standard")) standard_logistic_fit(form, d, predictor, model_id) else firth_fit(form, d, predictor, model_id)
  } else {
    out <- hc3_fit(form, d, predictor, model_id)
  }
  out$outcome <- outcome
  out$predictor <- predictor
  out$covariates <- paste(covars, collapse = "+")
  out
}

fit_manifest <- function(dat, manifest, binary_method = "firth") {
  rows <- vector("list", nrow(manifest))
  for (i in seq_len(nrow(manifest))) {
    m <- manifest[i, , drop = FALSE]
    covs <- if (is.list(manifest$covariates)) manifest$covariates[[i]] else character()
    if (!m$predictor %in% names(dat) || !m$outcome %in% names(dat)) {
      rows[[i]] <- data.frame(model_id = m$model_id, outcome = m$outcome, predictor = m$predictor, status = "MISSING_VARIABLE")
    } else {
      rows[[i]] <- fit_model(dat, m$outcome, m$predictor, covs, m$model_id, m$binary, if (m$binary) binary_method else NULL)
    }
    for (nm in setdiff(names(manifest), names(rows[[i]]))) rows[[i]][[nm]] <- manifest[[nm]][i]
  }
  bind_rows_base(rows)
}

bootstrap_model <- function(dat, outcome, predictor, covars, binary, reps, model_id) {
  d <- complete_model_data(dat, c(outcome, predictor, covars))
  n <- nrow(d)
  full <- fit_model(d, outcome, predictor, covars, model_id, binary)
  if (!nrow(full) || full$status[1] != "OK") return(data.frame(model_id = model_id, status = "FULL_FIT_FAILED"))
  vals <- rep(NA_real_, reps)
  if (binary) {
    y <- num(d[[outcome]])
    strata <- split(seq_len(n), y)
    for (b in seq_len(reps)) {
      idx <- unlist(lapply(strata, function(ii) sample(ii, length(ii), replace = TRUE)), use.names = FALSE)
      rr <- fit_model(d[idx, , drop = FALSE], outcome, predictor, covars, model_id, TRUE)
      if (nrow(rr) && rr$status[1] == "OK") vals[b] <- rr$estimate[1]
    }
  } else {
    for (b in seq_len(reps)) {
      idx <- sample.int(n, n, replace = TRUE)
      rr <- fit_model(d[idx, , drop = FALSE], outcome, predictor, covars, model_id, FALSE)
      if (nrow(rr) && rr$status[1] == "OK") vals[b] <- rr$estimate[1]
    }
  }
  vals <- vals[is.finite(vals)]
  data.frame(
    model_id = model_id,
    n = n,
    binary = binary,
    full_estimate = full$estimate[1],
    full_OR = if (binary) full$odds_ratio[1] else NA,
    bootstrap_reps_requested = reps,
    bootstrap_reps_valid = length(vals),
    bootstrap_low = if (length(vals)) unname(quantile(vals, 0.025)) else NA,
    bootstrap_high = if (length(vals)) unname(quantile(vals, 0.975)) else NA,
    bootstrap_OR_low = if (binary && length(vals)) exp(unname(quantile(vals, 0.025))) else NA,
    bootstrap_OR_high = if (binary && length(vals)) exp(unname(quantile(vals, 0.975))) else NA,
    status = ifelse(length(vals) >= 0.9 * reps, "OK", "LOW_VALID_BOOTSTRAPS")
  )
}

loo_manifest <- function(dat, manifest, apply_fdr_fun = NULL) {
  ids <- unique(as.character(dat$participant_id))
  rows <- parallel_lapply(seq_along(ids), function(i) {
    rr <- fit_manifest(
      dat[dat$participant_id != ids[i], , drop = FALSE],
      manifest,
      binary_method = "firth"
    )
    rr$omitted_id <- ids[i]
    if (!is.null(apply_fdr_fun)) rr <- apply_fdr_fun(rr)
    rr
  })
  bind_rows_base(rows)
}

summarise_loo <- function(full, loo, q_cols = character()) {
  rows <- list()
  for (mid in unique(full$model_id)) {
    f <- full[full$model_id == mid, , drop = FALSE]
    x <- loo[loo$model_id == mid, , drop = FALSE]
    if (!nrow(f) || !nrow(x)) next
    e <- num(x$estimate)
    p <- num(x$p_value)
    ef <- e[is.finite(e)]
    pf <- p[is.finite(p)]
    full_est <- num(f$estimate[1])
    row <- data.frame(
      model_id = mid,
      outcome = f$outcome[1],
      predictor = f$predictor[1],
      full_estimate = full_est,
      full_p = num(f$p_value[1]),
      loo_n = nrow(x),
      loo_estimate_median = if (length(ef)) median(ef) else NA,
      loo_estimate_min = if (length(ef)) min(ef) else NA,
      loo_estimate_max = if (length(ef)) max(ef) else NA,
      direction_retained_fraction = if (length(ef) && is.finite(full_est)) mean(sign(ef) == sign(full_est)) else NA,
      nominal_p_lt_0p05_fraction = if (length(pf)) mean(pf < 0.05) else NA,
      maximum_absolute_estimate_change = if (length(ef) && is.finite(full_est)) max(abs(ef - full_est)) else NA,
      stringsAsFactors = FALSE
    )
    for (qc in q_cols) {
      if (qc %in% names(x)) row[[paste0(qc, "_lt_0p05_fraction")]] <- mean(num(x[[qc]]) < 0.05, na.rm = TRUE)
    }
    rows[[length(rows) + 1L]] <- row
  }
  bind_rows_base(rows)
}

influence_for_manifest <- function(dat, manifest, family_label) {
  rows <- list()
  for (i in seq_len(nrow(manifest))) {
    m <- manifest[i, , drop = FALSE]
    if (m$binary || !m$predictor %in% names(dat) || !m$outcome %in% names(dat)) next
    covs <- manifest$covariates[[i]]
    d <- complete_model_data(dat, c("participant_id", m$outcome, m$predictor, covs))
    fit <- lm(as.formula(formula_string(m$outcome, c(m$predictor, covs))), data = d)
    db <- dfbeta(fit)
    dbp <- if (m$predictor %in% colnames(db)) db[, m$predictor] else rep(NA_real_, nrow(d))
    X <- model.matrix(fit)
    z <- data.frame(
      family = family_label,
      model_id = m$model_id,
      participant_id = d$participant_id,
      cooks_distance = cooks.distance(fit),
      leverage = hatvalues(fit),
      studentised_residual = rstudent(fit),
      dfbeta_predictor = dbp,
      stringsAsFactors = FALSE
    )
    z$flag_cook <- z$cooks_distance > 4 / nrow(d)
    z$flag_leverage <- z$leverage > 2 * ncol(X) / nrow(d)
    z$flag_studentised <- abs(z$studentised_residual) > 3
    z$flag_dfbeta <- abs(z$dfbeta_predictor) > 2 / sqrt(nrow(d))
    rows[[length(rows) + 1L]] <- z

    pdf(file.path(OUT, "plots", paste0(gsub("[^A-Za-z0-9]+", "_", m$model_id), "_diagnostics.pdf")), width = 10, height = 5)
    par(mfrow = c(1, 2))
    plot(fit, which = 1)
    plot(fit, which = 2)
    dev.off()
  }
  bind_rows_base(rows)
}

robust_for_manifest <- function(dat, manifest, family_label) {
  rows <- list()
  for (i in seq_len(nrow(manifest))) {
    m <- manifest[i, , drop = FALSE]
    if (m$binary || !m$predictor %in% names(dat) || !m$outcome %in% names(dat)) next
    covs <- manifest$covariates[[i]]
    d <- complete_model_data(dat, c(m$outcome, m$predictor, covs))
    r <- huber_fit(as.formula(formula_string(m$outcome, c(m$predictor, covs))), d, m$predictor, m$model_id)
    r$family <- family_label
    rows[[length(rows) + 1L]] <- r
  }
  bind_rows_base(rows)
}

loocv_binary <- function(dat, outcome, predictor, covars, model_id) {
  d <- complete_model_data(dat, c("participant_id", outcome, predictor, covars))
  y <- num(d[[outcome]])
  pred <- rep(NA_real_, nrow(d))
  for (i in seq_len(nrow(d))) {
    tr <- d[-i, , drop = FALSE]
    te <- d[i, , drop = FALSE]
    form <- as.formula(formula_string(outcome, c(predictor, covars)))
    mf <- model.frame(form, data = tr)
    yy <- model.response(mf)
    X <- model.matrix(form, mf)
    ff <- tryCatch(firth_matrix(X, yy), error = function(e) NULL)
    if (is.null(ff) || !ff$converged) next
    mm <- model.matrix(delete.response(terms(form)), te)
    common <- intersect(colnames(mm), colnames(X))
    eta <- sum(mm[1, common] * ff$beta[match(common, colnames(X))])
    pred[i] <- plogis(eta)
  }
  keep <- is.finite(pred)
  if (sum(keep) < 10) return(data.frame(model_id = model_id, status = "INSUFFICIENT_PREDICTIONS"))
  p <- pred[keep]
  yy <- y[keep]
  calibration <- tryCatch(glm(yy ~ qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6)), family = binomial()), error = function(e) NULL)
  data.frame(
    model_id = model_id,
    n = length(yy),
    events = sum(yy == 1),
    AUC = auc_rank(yy, p),
    Brier = mean((p - yy)^2),
    calibration_intercept = if (!is.null(calibration)) coef(calibration)[1] else NA,
    calibration_slope = if (!is.null(calibration)) coef(calibration)[2] else NA,
    predicted_min = min(p),
    predicted_max = max(p),
    status = "OK"
  )
}

prepare_dataset <- function(dat, label, pc2_lookup = NULL) {
  pid <- resolve_col(dat, c("participant_id", "subject_id", "participant"), required = TRUE, label = paste(label, "participant ID"))
  dat$participant_id <- canon_pid(dat[[pid]])
  group <- resolve_col(dat, c("group", "clinical_group"), required = FALSE)
  cpsp <- resolve_col(dat, c("CPSP_status_binary", ".CPSP_binary", "CPSP_binary", "CPSP_status"), required = FALSE)
  if (!is.na(cpsp)) {
    x <- num(dat[[cpsp]])
    if (all(is.na(x))) {
      z <- toupper(trimws(as.character(dat[[cpsp]])))
      dat$CPSP_status_binary <- as.integer(z %in% c("CPSP", "YES", "TRUE", "CASE", "POSITIVE"))
    } else {
      dat$CPSP_status_binary <- as.integer(x > 0)
    }
  } else if (!is.na(group)) {
    dat$CPSP_status_binary <- as.integer(toupper(trimws(as.character(dat[[group]]))) == "CPSP")
  } else {
    stop("Could not derive CPSP status in ", label, call. = FALSE)
  }

  npsi <- resolve_col(dat, c("NPSI_total", "NPSI_total_model", "NPSI"), required = TRUE, label = paste(label, "NPSI total"))
  pc2 <- resolve_col(dat, c("QST_PC2_model", "QST_PC2", "qst_pc2", "PC2"), required = FALSE)
  pc1 <- resolve_col(dat, c("QST_PC1_model", "QST_PC1", "qst_pc1", "PC1"), required = TRUE, label = paste(label, "QST PC1"))
  dat$NPSI_total_four <- num(dat[[npsi]])
  if (!is.na(pc2)) {
    dat$QST_PC2_four <- num(dat[[pc2]])
    pc2_source <- pc2
  } else if (!is.null(pc2_lookup)) {
    dat$QST_PC2_four <- unname(pc2_lookup[dat$participant_id])
    pc2_source <- "participant_id lookup from main dataset"
  } else {
    stop(
      "Could not resolve saved participant-level QST PC2 in ", label,
      ". Expected one of QST_PC2_model, QST_PC2, qst_pc2 or PC2.",
      call. = FALSE
    )
  }
  dat$QST_PC1_four <- num(dat[[pc1]])

  volz <- resolve_col(dat, c("z_log1p_lesion_volume", "z_reviewer_lesion_volume"), required = FALSE)
  if (is.na(volz)) {
    logvol <- resolve_col(dat, c("log1p_lesion_volume"), required = FALSE)
    rawvol <- resolve_col(dat, c("lesion_volume_mm3_primary_BranchB", "lesion_volume_mm3", "lesion_volume"), required = FALSE)
    if (!is.na(logvol)) dat$z_log1p_lesion_volume_four <- safe_z(dat[[logvol]])
    else if (!is.na(rawvol)) dat$z_log1p_lesion_volume_four <- safe_z(log1p(num(dat[[rawvol]])))
    else stop("Could not derive lesion-volume covariate in ", label, call. = FALSE)
    volz <- "z_log1p_lesion_volume_four"
  }

  side <- resolve_col(dat, c("original_lesion_side_model", "original_lesion_side", "lesion_side"), required = TRUE, label = paste(label, "lesion side"))
  dat[[side]] <- factor(dat[[side]])

  age <- resolve_col(dat, c("z_age", "age", "Age", "age_years"), required = FALSE)
  if (!is.na(age) && !grepl("^z_", age)) {
    dat$z_age_four <- safe_z(dat[[age]])
    age <- "z_age_four"
  }
  sex <- resolve_col(dat, c("sex", "gender"), required = FALSE)
  if (!is.na(sex)) dat[[sex]] <- factor(dat[[sex]])

  chronic <- resolve_col(dat, c("z_log1p_months_since_stroke_manual", "z_log1p_months_since_stroke", "z_months_since_stroke"), required = FALSE)
  if (is.na(chronic)) {
    months <- resolve_col(dat, c("months_since_stroke_manual", "months_since_stroke", "Months since stroke"), required = FALSE)
    if (!is.na(months)) {
      dat$z_log1p_months_since_stroke_four <- safe_z(log1p(pmax(num(dat[[months]]), 0)))
      chronic <- "z_log1p_months_since_stroke_four"
    }
  }

  attr(dat, "volz") <- volz
  attr(dat, "side") <- side
  attr(dat, "age") <- age
  attr(dat, "sex") <- sex
  attr(dat, "chronic") <- chronic
  attr(dat, "pc2_source") <- pc2_source
  dat
}

main <- prepare_dataset(read_any(MAIN_DATA), "main")
pc2_lookup <- setNames(main$QST_PC2_four, main$participant_id)
s1 <- if (!is.na(S1_DATA)) prepare_dataset(read_any(S1_DATA), "S1", pc2_lookup) else NULL
morel <- if (!is.na(MOREL_DATA)) prepare_dataset(read_any(MOREL_DATA), "Morel", pc2_lookup) else NULL

heat_flag <- resolve_col(main, c("heat_only_all_branches_replacement"), required = TRUE, label = "heat-only all-branch replacement flag")
if (nrow(main) != 63L || any(num(main[[heat_flag]]) != 1L, na.rm = TRUE) || any(!is.finite(num(main[[heat_flag]])))) {
  stop("Main dataset is not a complete 63-participant heat-only all-branch replacement dataset", call. = FALSE)
}
heat_definition <- resolve_col(main, c("primary_task_roi_definition"), required = TRUE, label = "heat-only primary ROI definition")
if (length(unique(as.character(main[[heat_definition]]))) != 1L || !grepl("heatOnly", unique(as.character(main[[heat_definition]])), fixed = TRUE)) {
  stop("Primary ROI definition does not identify the heat-only ROI", call. = FALSE)
}
heat_scope <- resolve_col(main, c("task_roi_replacement_scope"), required = TRUE, label = "heat-only branch replacement scope")
expected_heat_scope <- c("LNM_BranchB", "LNM_BranchA", "BCB_BranchB", "BCB_BranchA", "BCB_BranchC")
observed_heat_scope <- unique(strsplit(as.character(main[[heat_scope]][1]), ";", fixed = TRUE)[[1]])
if (!setequal(observed_heat_scope, expected_heat_scope)) {
  stop("Heat-only replacement scope does not contain the required five branches", call. = FALSE)
}

OUTCOME_META <- data.frame(
  outcome = c("CPSP_status_binary", "NPSI_total_four", "QST_PC2_four", "QST_PC1_four"),
  outcome_label = c("CPSP_status", "NPSI_total", "QST_PC2", "QST_PC1"),
  domain = c("clinical", "clinical", "QST", "QST"),
  binary = c(TRUE, FALSE, FALSE, FALSE),
  stringsAsFactors = FALSE
)
write_tsv(OUTCOME_META, "four_outcome_definition.tsv", "audit")

make_covariates <- function(dat) {
  base <- c(attr(dat, "volz"), attr(dat, "side"))
  age_sex <- c(base, attr(dat, "age"), attr(dat, "sex"))
  age_sex <- age_sex[!is.na(age_sex) & nzchar(age_sex)]
  chronic <- c(base, attr(dat, "chronic"))
  chronic <- chronic[!is.na(chronic) & nzchar(chronic)]
  age_sex_chronic <- unique(c(age_sex, attr(dat, "chronic")))
  age_sex_chronic <- age_sex_chronic[!is.na(age_sex_chronic) & nzchar(age_sex_chronic)]
  list(base = base, age_sex = age_sex, chronic = chronic, age_sex_chronic = age_sex_chronic)
}

main_cov <- make_covariates(main)
s1_cov <- if (!is.null(s1)) make_covariates(s1) else NULL
morel_cov <- if (!is.null(morel)) make_covariates(morel) else NULL

resolve_predictor <- function(dat, z_aliases, raw_aliases = character(), regex = NULL, z_output = NULL) {
  z <- resolve_col(dat, z_aliases, regex = regex, required = FALSE)
  if (!is.na(z)) return(list(dat = dat, predictor = z))
  raw <- resolve_col(dat, raw_aliases, regex = regex, required = FALSE)
  if (is.na(raw)) return(list(dat = dat, predictor = NA_character_))
  if (is.null(z_output)) z_output <- paste0("z__four__", gsub("[^A-Za-z0-9]+", "_", raw))
  dat[[z_output]] <- safe_z(dat[[raw]])
  list(dat = dat, predictor = z_output)
}

pr <- resolve_predictor(main,
  c("z_LNM_BA3a_BA3c_primary_BranchB"),
  c("LNM_BA3a_BA3c_primary_BranchB"),
  regex = "^z.*LNM.*BA3a.*BA3c.*BranchB$",
  z_output = "z_LNM_BA3a_BA3c_primary_BranchB"
)
main <- pr$dat; P_LNM_B <- pr$predictor
pr <- resolve_predictor(main,
  c("z_BCB_BA3a_BA3c_primary_BranchB_2mm"),
  c("BCB_BA3a_BA3c_primary_BranchB_2mm"),
  regex = "^z.*BCB.*BA3a.*BA3c.*BranchB.*2mm$",
  z_output = "z_BCB_BA3a_BA3c_primary_BranchB_2mm"
)
main <- pr$dat; P_BCB_B <- pr$predictor

expand_predictors_outcomes <- function(predictors, outcomes = OUTCOME_META, family_prefix, covars, adjustment = "minimal") {
  rows <- list()
  k <- 0L
  for (i in seq_len(nrow(predictors))) {
    for (j in seq_len(nrow(outcomes))) {
      k <- k + 1L
      rows[[k]] <- data.frame(
        model_id = paste(family_prefix, predictors$roi[i], predictors$variant[i], predictors$modality[i], outcomes$outcome_label[j], adjustment, sep = "__"),
        analysis_family = family_prefix,
        roi = predictors$roi[i],
        variant = predictors$variant[i],
        modality = predictors$modality[i],
        branch = predictors$branch[i],
        outcome = outcomes$outcome[j],
        outcome_label = outcomes$outcome_label[j],
        domain = outcomes$domain[j],
        binary = outcomes$binary[j],
        predictor = predictors$predictor[i],
        adjustment = adjustment,
        stringsAsFactors = FALSE
      )
      rows[[k]]$covariates <- I(list(covars))
    }
  }
  bind_rows_base(rows)
}

apply_primary_fdr <- function(x) {
  if (!nrow(x)) return(x)
  p <- num(x$p_value)
  x$q_within_modality_4 <- bh_within(p, paste(x$modality, x$branch, x$variant, x$adjustment, sep = "__"))
  x$holm_within_modality_4 <- holm_within(p, paste(x$modality, x$branch, x$variant, x$adjustment, sep = "__"))
  x$q_combined_modalities_8 <- bh_within(p, paste(x$branch, x$variant, x$adjustment, sep = "__"))
  x
}

apply_structured_fdr <- function(x) {
  if (!nrow(x)) return(x)
  p <- num(x$p_value)
  x$q_domain_modality_6 <- bh_within(p, paste(x$domain, x$modality, x$branch, x$variant, x$adjustment, sep = "__"))
  x$q_all4_modality_12 <- bh_within(p, paste(x$modality, x$branch, x$variant, x$adjustment, sep = "__"))
  x$q_domain_combined_modalities_12 <- bh_within(p, paste(x$domain, x$branch, x$variant, x$adjustment, sep = "__"))
  x$q_all4_combined_modalities_24 <- bh_within(p, paste(x$branch, x$variant, x$adjustment, sep = "__"))
  x
}

apply_s1_fdr <- function(x) {
  if (!nrow(x)) return(x)
  p <- num(x$p_value)
  x$q_within_roi_variant_modality_4 <- bh_within(p, paste(x$roi, x$variant, x$modality, x$branch, x$adjustment, sep = "__"))
  x$holm_within_roi_variant_modality_4 <- holm_within(p, paste(x$roi, x$variant, x$modality, x$branch, x$adjustment, sep = "__"))
  x$q_combined_modalities_within_roi_variant_8 <- bh_within(p, paste(x$roi, x$variant, x$branch, x$adjustment, sep = "__"))
  x
}

apply_morel_percentage_fdr <- function(x) {
  if (!nrow(x)) return(x)
  x$q_Morel_percentage_four <- p.adjust(num(x$p_value), method = "BH")
  x$Holm_Morel_percentage_four <- p.adjust(num(x$p_value), method = "holm")
  x
}

apply_localisation_fdr <- function(x) {
  if (!nrow(x)) return(x)
  x$q_localisation_all_four <- p.adjust(num(x$p_value), method = "BH")
  x$Holm_localisation_all_four <- p.adjust(num(x$p_value), method = "holm")
  x$q_localisation_by_modality <- bh_within(num(x$p_value), paste(x$modality, x$adjustment, sep = "__"))
  x
}

flexible_volume_models <- function(dat, manifest, label) {
  rows <- list()
  vol <- attr(dat, "volz")
  side <- attr(dat, "side")
  for (i in seq_len(nrow(manifest))) {
    m <- manifest[i, , drop = FALSE]
    specs <- list(
      linear = c(m$predictor, vol, side),
      quadratic = c(m$predictor, vol, paste0("I(", vol, "^2)"), side),
      spline3 = c(m$predictor, paste0("splines::ns(", vol, ", df=3)"), side),
      interaction = c(m$predictor, vol, paste0(m$predictor, ":", vol), side)
    )
    for (sn in names(specs)) {
      form <- as.formula(formula_string(m$outcome, specs[[sn]]))
      d <- complete_model_data(dat, c(m$outcome, m$predictor, vol, side))
      model_id <- paste(label, m$model_id, sn, sep = "__")
      r_main <- if (m$binary) firth_fit(form, d, m$predictor, model_id) else hc3_fit(form, d, m$predictor, model_id)
      r_main$outcome <- m$outcome
      r_main$predictor <- m$predictor
      r_main$volume_model <- sn
      r_main$reported_term_role <- "imaging_main_effect"
      rows[[length(rows) + 1L]] <- r_main
      if (sn == "interaction") {
        int_term <- paste0(m$predictor, ":", vol)
        r_int <- if (m$binary) firth_fit(form, d, int_term, paste0(model_id, "__interaction_term")) else hc3_fit(form, d, int_term, paste0(model_id, "__interaction_term"))
        r_int$outcome <- m$outcome
        r_int$predictor <- m$predictor
        r_int$volume_model <- sn
        r_int$reported_term_role <- "imaging_by_lesion_volume_interaction"
        rows[[length(rows) + 1L]] <- r_int
      }
    }
  }
  bind_rows_base(rows)
}

primary_predictors <- data.frame(
  roi = "BA3a_BA3c_heat_only_task_defined",
  variant = "BranchB_heat_only_primary",
  modality = c("LNM", "BCB"),
  branch = "BranchB",
  predictor = c(P_LNM_B, P_BCB_B),
  stringsAsFactors = FALSE
)
primary_predictors <- primary_predictors[!is.na(primary_predictors$predictor), , drop = FALSE]
primary_manifest <- expand_predictors_outcomes(primary_predictors, family_prefix = "primary_BA3a_BA3c_heat_only_four", covars = main_cov$base)
primary_full <- apply_primary_fdr(fit_manifest(main, primary_manifest))
write_tsv(primary_manifest, "primary_BA3a_BA3c_four_manifest.tsv")
write_tsv(primary_full, "primary_BA3a_BA3c_four_full.tsv")

primary_cpsp_manifest <- primary_manifest[primary_manifest$binary, , drop = FALSE]
primary_standard_logistic <- fit_manifest(main, primary_cpsp_manifest, binary_method = "standard")
write_tsv(primary_standard_logistic, "primary_BA3a_BA3c_standard_logistic_companion.tsv")

primary_sens <- list()
for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
  cv <- main_cov[[nm]]
  if (length(cv) <= length(main_cov$base)) next
  man <- expand_predictors_outcomes(primary_predictors, family_prefix = "primary_BA3a_BA3c_heat_only_four", covars = cv, adjustment = nm)
  primary_sens[[nm]] <- apply_primary_fdr(fit_manifest(main, man))
}
primary_sensitivity <- bind_rows_base(primary_sens)
write_tsv(primary_sensitivity, "primary_BA3a_BA3c_age_sex_chronicity_sensitivities.tsv")

branch_defs <- list(
  list(roi = "BA3a_BA3c_heat_only_task_defined", variant = "BranchA_heat_only", modality = "LNM", branch = "BranchA", z = c("z_LNM_BA3a_BA3c_sensitivity_BranchA"), raw = c("LNM_BA3a_BA3c_sensitivity_BranchA"), regex = "^z_LNM_BA3a_BA3c_sensitivity_BranchA$"),
  list(roi = "BA3a_BA3c_heat_only_task_defined", variant = "BranchA_heat_only", modality = "BCB", branch = "BranchA", z = c("z_BCB_BA3a_BA3c_sensitivity_BranchA_2mm"), raw = c("BCB_BA3a_BA3c_sensitivity_BranchA_2mm"), regex = "^z_BCB_BA3a_BA3c_sensitivity_BranchA_2mm$"),
  list(roi = "BA3a_BA3c_heat_only_task_defined", variant = "BranchC_heat_only", modality = "BCB", branch = "BranchC", z = c("z_BCB_BA3a_BA3c_sensitivity_BranchC_1mm"), raw = c("BCB_BA3a_BA3c_sensitivity_BranchC_1mm"), regex = "^z_BCB_BA3a_BA3c_sensitivity_BranchC_1mm$")
)
branch_pred_rows <- list()
for (i in seq_along(branch_defs)) {
  d <- branch_defs[[i]]
  rr <- resolve_predictor(main, d$z, d$raw, d$regex)
  main <- rr$dat
  if (is.na(rr$predictor)) stop("Required heat-only branch predictor is missing: ", d$branch, " ", d$modality, call. = FALSE)
  branch_pred_rows[[length(branch_pred_rows) + 1L]] <- data.frame(
    roi = d$roi, variant = d$variant, modality = d$modality,
    branch = d$branch, predictor = rr$predictor, stringsAsFactors = FALSE
  )
}
primary_branch_predictors <- bind_rows_base(branch_pred_rows)
if (nrow(primary_branch_predictors) != 3L) stop("Expected exactly three heat-only branch-sensitivity predictors", call. = FALSE)
primary_branch_manifest <- expand_predictors_outcomes(primary_branch_predictors, family_prefix = "primary_heat_only_branch_sensitivity_four", covars = main_cov$base)
primary_branch_results <- apply_primary_fdr(fit_manifest(main, primary_branch_manifest))
write_tsv(primary_branch_manifest, "primary_BA3a_BA3c_branch_sensitivities_manifest.tsv")
write_tsv(primary_branch_results, "primary_BA3a_BA3c_branch_sensitivities_four.tsv")
write_tsv(data.frame(
  analysis = c("LNM Branch A", "BCB Branch A", "BCB Branch C"),
  included = TRUE,
  ROI_definition = "HC19 heat-only left S1 plus BA3c-transition, Z > 2.3",
  purpose = c(
    "fMRIPrep weighted-PV functional LNM sensitivity",
    "fMRIPrep 2 mm structural-disconnectome sensitivity",
    "enantiomorphic 1 mm structural-disconnectome sensitivity"
  ),
  stringsAsFactors = FALSE
), "HEAT_ONLY_PRIMARY_BRANCH_SENSITIVITY_SCOPE.tsv", "audit")

exclusion_candidates <- list(
  exclude_top5 = c("sensitivity_exclude_large_lesion_top5", "large_lesion_top5_flag"),
  exclude_top10 = c("sensitivity_exclude_large_lesion_top10", "large_lesion_top10_flag"),
  exclude_high_motion = c("sensitivity_exclude_high_motion", "high_motion_exclusion_flag", "high_motion_sensitivity_flag"),
  exclude_low_LNM_coverage = c("sensitivity_exclude_low_LNM_coverage", "low_LNM_coverage_flag"),
  exclude_complex_bilateral = c("sensitivity_exclude_complex_bilateral", "complex_or_bilateral_lesion_flag", "complex_or_qc_flag"),
  exclude_DN4_positive_NN_strict_control = c("NN_DN4_positive", "strict_control_exclusion_flag", "discordant_NN_flag")
)
exclusion_rows <- list()
for (nm in names(exclusion_candidates)) {
  fc <- resolve_col(main, exclusion_candidates[[nm]], required = FALSE)
  if (is.na(fc)) next
  flag <- num(main[[fc]])
  keep <- is.na(flag) | flag == 0
  rr <- apply_primary_fdr(fit_manifest(main[keep, , drop = FALSE], primary_manifest))
  rr$sensitivity <- nm
  rr$excluded_n <- sum(!keep)
  exclusion_rows[[length(exclusion_rows) + 1L]] <- rr
}
write_tsv(bind_rows_base(exclusion_rows), "primary_BA3a_BA3c_sample_exclusion_sensitivities_four.tsv")

write_tsv(
  flexible_volume_models(main, primary_manifest, "primary_BA3a_BA3c_heat_only"),
  "primary_BA3a_BA3c_flexible_lesion_volume_models_four.tsv"
)

structured_defs <- expand.grid(
  roi = c("whole_thalamus", "operculo_insular_S2_composite", "cingulate_broad_exploratory"),
  modality = c("LNM", "BCB"),
  stringsAsFactors = FALSE
)
structured_defs$branch <- "BranchB"
structured_defs$variant <- "BranchB_primary"
structured_defs$predictor <- NA_character_
for (i in seq_len(nrow(structured_defs))) {
  roi <- structured_defs$roi[i]
  mod <- structured_defs$modality[i]
  zname <- paste0("z_ROI_", roi, "_", mod, "_BranchB")
  raw <- paste0("ROI_", roi, "_", mod, "_BranchB")
  rr <- resolve_predictor(main, c(zname), c(raw), regex = paste0("^z.*", gsub("_", ".*", roi), ".*", mod, ".*BranchB$"), z_output = zname)
  main <- rr$dat
  structured_defs$predictor[i] <- rr$predictor
}
structured_defs <- structured_defs[!is.na(structured_defs$predictor), , drop = FALSE]
structured_manifest <- expand_predictors_outcomes(structured_defs, family_prefix = "structured_secondary_four", covars = main_cov$base)
structured_full <- apply_structured_fdr(fit_manifest(main, structured_manifest))
write_tsv(structured_manifest, "structured_secondary_24_manifest.tsv")
write_tsv(structured_full, "structured_secondary_24_full.tsv")

structured_sens <- list()
for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
  cv <- main_cov[[nm]]
  if (length(cv) <= length(main_cov$base)) next
  man <- expand_predictors_outcomes(structured_defs, family_prefix = "structured_secondary_four", covars = cv, adjustment = nm)
  structured_sens[[nm]] <- apply_structured_fdr(fit_manifest(main, man))
}
write_tsv(bind_rows_base(structured_sens), "structured_secondary_age_sex_chronicity_sensitivities_four.tsv")

structured_operculo_manifest <- structured_manifest[structured_manifest$roi == "operculo_insular_S2_composite", , drop = FALSE]
write_tsv(
  flexible_volume_models(main, structured_operculo_manifest, "structured_operculo"),
  "structured_operculo_flexible_lesion_volume_models_four.tsv"
)

structured_branch_rows <- list()
for (roi in unique(structured_defs$roi)) {
  for (mod in c("LNM", "BCB")) {
    branches <- if (mod == "LNM") c("BranchA") else c("BranchA", "BranchC")
    for (br in branches) {
      zname <- paste0("z_ROI_", roi, "_", mod, "_", br)
      raw <- paste0("ROI_", roi, "_", mod, "_", br)
      rr <- resolve_predictor(main, c(zname), c(raw), regex = paste0("^z.*", gsub("_", ".*", roi), ".*", mod, ".*", br, "$"), z_output = zname)
      main <- rr$dat
      if (!is.na(rr$predictor)) structured_branch_rows[[length(structured_branch_rows) + 1L]] <- data.frame(roi = roi, modality = mod, branch = br, variant = br, predictor = rr$predictor)
    }
  }
}
structured_branch_defs <- bind_rows_base(structured_branch_rows)
structured_branch_manifest <- if (nrow(structured_branch_defs)) expand_predictors_outcomes(structured_branch_defs, family_prefix = "structured_branch_sensitivity_four", covars = main_cov$base) else data.frame()
structured_branch_results <- if (nrow(structured_branch_manifest)) apply_structured_fdr(fit_manifest(main, structured_branch_manifest)) else data.frame()
write_tsv(structured_branch_results, "structured_secondary_branch_sensitivities_four.tsv")

structured_exclusion_rows <- list()
for (nm in names(exclusion_candidates)) {
  fc <- resolve_col(main, exclusion_candidates[[nm]], required = FALSE)
  if (is.na(fc)) next
  flag <- num(main[[fc]])
  keep <- is.na(flag) | flag == 0
  rr <- apply_structured_fdr(fit_manifest(main[keep, , drop = FALSE], structured_manifest))
  rr$sensitivity <- nm
  rr$excluded_n <- sum(!keep)
  structured_exclusion_rows[[length(structured_exclusion_rows) + 1L]] <- rr
}
write_tsv(bind_rows_base(structured_exclusion_rows), "structured_secondary_sample_exclusion_sensitivities_four.tsv")

loc_patterns <- c("aMCC", "dACC", "ACC_proxy", "anterior_cingulate", "HCPex.*operculo", "HCPex.*S2")
loc_cols <- unique(unlist(lapply(loc_patterns, function(p) grep(p, names(main), ignore.case = TRUE, value = TRUE))))
loc_cols <- loc_cols[grepl("^z", loc_cols, ignore.case = TRUE)]
loc_cols <- loc_cols[!grepl("BranchA|BranchC", loc_cols, ignore.case = TRUE)]
loc_defs <- list()
for (col in loc_cols) {
  mod <- if (grepl("LNM", col, ignore.case = TRUE)) "LNM" else if (grepl("BCB", col, ignore.case = TRUE)) "BCB" else NA_character_
  if (is.na(mod)) next
  roi <- gsub("^z[_]*", "", col)
  loc_defs[[length(loc_defs) + 1L]] <- data.frame(roi = roi, modality = mod, branch = "BranchB", variant = "localisation", predictor = col)
}
loc_defs <- bind_rows_base(loc_defs)
localisation_manifest <- if (nrow(loc_defs)) expand_predictors_outcomes(loc_defs, family_prefix = "HCPex_localisation_four", covars = main_cov$base) else data.frame()
localisation_full <- if (nrow(localisation_manifest)) apply_localisation_fdr(fit_manifest(main, localisation_manifest)) else data.frame()
write_tsv(localisation_full, "HCPex_localisation_four_full.tsv")

localisation_sens_rows <- list()
if (nrow(loc_defs)) {
  for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
    cv <- main_cov[[nm]]
    if (length(cv) <= length(main_cov$base)) next
    man <- expand_predictors_outcomes(loc_defs, family_prefix = "HCPex_localisation_four", covars = cv, adjustment = nm)
    localisation_sens_rows[[length(localisation_sens_rows) + 1L]] <- apply_localisation_fdr(fit_manifest(main, man))
  }
}
write_tsv(bind_rows_base(localisation_sens_rows), "HCPex_localisation_age_sex_chronicity_sensitivities_four.tsv")

metric_suffixes <- c(
  conditional_mean_nonzero = "conditional_intensity_nonzero_mean",
  mean_all = "all_ROI_mean",
  extent_pct = "percentage_ROI_affected",
  sum_probability = "integrated_probability_burden",
  p95 = "upper_quantile_p95",
  maximum = "maximum_probability"
)
alt_rows <- list()
for (suffix in names(metric_suffixes)) {
  raw <- paste0("BCB_operculo_insular_S2_historical_original_", suffix)
  if (!raw %in% names(main)) next
  zpred <- paste0("z__four__", raw)
  main[[zpred]] <- safe_z(main[[raw]])
  for (j in seq_len(nrow(OUTCOME_META))) {
    o <- OUTCOME_META[j, ]
    r <- fit_model(main, o$outcome, zpred, main_cov$base, paste("operculo_alt_BCB", suffix, o$outcome_label, sep = "__"), o$binary)
    r$metric <- suffix
    r$metric_interpretation <- metric_suffixes[[suffix]]
    r$outcome_label <- o$outcome_label
    r$domain <- o$domain
    alt_rows[[length(alt_rows) + 1L]] <- r
  }
}
alt_results <- bind_rows_base(alt_rows)
if (nrow(alt_results)) {
  alt_results$q_four_outcomes_within_metric <- bh_within(num(alt_results$p_value), alt_results$metric)
  alt_results$q_across_metrics_within_outcome <- bh_within(num(alt_results$p_value), alt_results$outcome_label)
  alt_results$q_across_all_metric_outcome_models <- p.adjust(num(alt_results$p_value), "BH")
}
write_tsv(alt_results, "operculo_BCB_alternative_estimands_four.tsv")

joint_rows <- list()
extent_raw <- "BCB_operculo_insular_S2_historical_original_extent_pct"
intensity_raw <- "BCB_operculo_insular_S2_historical_original_conditional_mean_nonzero"
if (all(c(extent_raw, intensity_raw) %in% names(main))) {
  main$z__four__operculo_extent <- safe_z(main[[extent_raw]])
  main$z__four__operculo_intensity <- safe_z(main[[intensity_raw]])
  zext <- "z__four__operculo_extent"
  zint <- "z__four__operculo_intensity"
  for (j in seq_len(nrow(OUTCOME_META))) {
    o <- OUTCOME_META[j, ]
    d <- complete_model_data(main, c(o$outcome, zext, zint, main_cov$base))
    form <- as.formula(formula_string(o$outcome, c(zext, zint, main_cov$base)))
    if (o$binary) {
      r1 <- firth_fit(form, d, zext, paste("operculo_joint", o$outcome_label, "extent", sep = "__"))
      r2 <- firth_fit(form, d, zint, paste("operculo_joint", o$outcome_label, "intensity", sep = "__"))
    } else {
      r1 <- hc3_fit(form, d, zext, paste("operculo_joint", o$outcome_label, "extent", sep = "__"))
      r2 <- hc3_fit(form, d, zint, paste("operculo_joint", o$outcome_label, "intensity", sep = "__"))
    }
    for (r in list(r1, r2)) {
      r$outcome_label <- o$outcome_label
      r$extent_intensity_correlation <- cor(d[[zext]], d[[zint]])
      r$pairwise_VIF <- 1 / (1 - cor(d[[zext]], d[[zint]])^2)
      r$condition_number <- kappa(model.matrix(form, d))
      joint_rows[[length(joint_rows) + 1L]] <- r
    }
  }
}
write_tsv(bind_rows_base(joint_rows), "operculo_BCB_joint_extent_intensity_four.tsv")

main$NPSI_total_presence_four <- as.integer(main$NPSI_total_four > 0)
npsi_predictors <- unique(c(P_LNM_B, P_BCB_B, structured_defs$predictor[structured_defs$roi == "operculo_insular_S2_composite"], structured_defs$predictor[structured_defs$roi == "whole_thalamus"]))
npsi_predictors <- npsi_predictors[!is.na(npsi_predictors) & npsi_predictors %in% names(main)]
npsi_dist_rows <- list()
for (pred in npsi_predictors) {
  npsi_dist_rows[[length(npsi_dist_rows) + 1L]] <- fit_model(main, "NPSI_total_presence_four", pred, main_cov$base, paste("NPSI_presence", pred, sep = "__"), TRUE)
  pos <- main[main$NPSI_total_four > 0, , drop = FALSE]
  if (nrow(pos) >= 20) npsi_dist_rows[[length(npsi_dist_rows) + 1L]] <- fit_model(pos, "NPSI_total_four", pred, main_cov$base, paste("NPSI_positive_only", pred, sep = "__"), FALSE)
  cpsp_only <- main[main$CPSP_status_binary == 1, , drop = FALSE]
  if (nrow(cpsp_only) >= 20) npsi_dist_rows[[length(npsi_dist_rows) + 1L]] <- fit_model(cpsp_only, "NPSI_total_four", pred, main_cov$base, paste("NPSI_CPSP_only", pred, sep = "__"), FALSE)
}
write_tsv(bind_rows_base(npsi_dist_rows), "NPSI_total_distributional_sensitivities_four.tsv")

laterality_cols <- grep("BA3a.*BA3c.*(left|right|bilateral|ipsilesional|contralesional)|operculo.*(left|right|bilateral|ipsilesional|contralesional)", names(main), ignore.case = TRUE, value = TRUE)
laterality_cols <- laterality_cols[grepl("^z", laterality_cols, ignore.case = TRUE)]
laterality_rows <- list()
for (pred in laterality_cols) {
  for (j in seq_len(nrow(OUTCOME_META))) {
    o <- OUTCOME_META[j, ]
    laterality_rows[[length(laterality_rows) + 1L]] <- fit_model(main, o$outcome, pred, main_cov$base, paste("laterality", pred, o$outcome_label, sep = "__"), o$binary)
  }
}
laterality_results <- bind_rows_base(laterality_rows)
if (nrow(laterality_results)) laterality_results$q_BH_all_laterality_models <- p.adjust(num(laterality_results$p_value), "BH")
write_tsv(laterality_results, "laterality_and_alignment_models_four.tsv")

qc_covars <- c("complex_or_qc_flag", "complex_or_bilateral_lesion_flag", "high_motion_sensitivity_flag", "low_LNM_coverage_flag")
qc_covars <- qc_covars[qc_covars %in% names(main)]
qc_adjust_rows <- list()
headline_manifest <- bind_rows_base(list(primary_manifest, structured_manifest[structured_manifest$roi == "operculo_insular_S2_composite", , drop = FALSE]))
for (qc in qc_covars) {
  man <- headline_manifest
  man$adjustment <- paste0("plus_", qc)
  man$covariates <- I(rep(list(unique(c(main_cov$base, qc))), nrow(man)))
  rr <- fit_manifest(main, man)
  rr$qc_covariate <- qc
  qc_adjust_rows[[length(qc_adjust_rows) + 1L]] <- rr
}
write_tsv(bind_rows_base(qc_adjust_rows), "technical_anatomical_QC_adjustment_models_four.tsv")

s1_manifest <- data.frame()
s1_full <- data.frame()
s1_loo <- data.frame()
if (!is.null(s1)) {
  s1_metric_cols <- grep("^z__S1_(HCPex|Juelich)_", names(s1), value = TRUE)
  if (!length(s1_metric_cols)) s1_metric_cols <- grep("^z__S1_", names(s1), value = TRUE)
  parse_s1 <- function(col) {
    mod <- if (grepl("LNM", col)) "LNM" else if (grepl("BCB", col)) "BCB" else NA_character_
    branch <- if (grepl("BranchA", col)) "BranchA" else if (grepl("BranchC", col)) "BranchC" else "BranchB"
    roi <- if (grepl("BA1_BA3b_union", col)) "BA1_BA3b_union" else if (grepl("BA3b", col)) "BA3b" else if (grepl("BA3a", col)) "BA3a_atlas_only" else if (grepl("BA1", col)) "BA1" else NA_character_
    variant <- if (grepl("bilateral", col)) "bilateral" else if (grepl("ipsilesional", col)) "ipsilesional" else if (grepl("contralesional", col)) "contralesional" else if (grepl("right", col)) "right" else if (grepl("left", col)) "left" else "unspecified"
    data.frame(roi = roi, modality = mod, branch = branch, variant = variant, predictor = col)
  }
  defs <- bind_rows_base(lapply(s1_metric_cols, parse_s1))
  defs <- defs[!is.na(defs$roi) & !is.na(defs$modality), , drop = FALSE]
  main_defs <- defs[defs$branch == "BranchB" & defs$variant %in% c("left", "bilateral", "ipsilesional", "contralesional"), , drop = FALSE]
  main_defs <- main_defs[grepl("mean_fisher_z|conditional_nonzero_mean", main_defs$predictor), , drop = FALSE]
  main_defs <- main_defs[!duplicated(paste(main_defs$roi, main_defs$variant, main_defs$modality)), , drop = FALSE]
  if (nrow(main_defs)) {
    s1_manifest <- expand_predictors_outcomes(main_defs, family_prefix = "S1_Juelich_four", covars = s1_cov$base)
    s1_full <- apply_s1_fdr(fit_manifest(s1, s1_manifest))
    write_tsv(s1_manifest, "S1_Juelich_four_manifest.tsv")
    write_tsv(s1_full, "S1_Juelich_four_full.tsv")

    s1_sens <- list()
    for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
      cv <- s1_cov[[nm]]
      if (length(cv) <= length(s1_cov$base)) next
      man <- expand_predictors_outcomes(main_defs, family_prefix = "S1_Juelich_four", covars = cv, adjustment = nm)
      s1_sens[[nm]] <- apply_s1_fdr(fit_manifest(s1, man))
    }
    write_tsv(bind_rows_base(s1_sens), "S1_Juelich_age_sex_chronicity_sensitivities_four.tsv")

    s1_headline_left <- s1_manifest[s1_manifest$roi %in% c("BA1", "BA3b", "BA1_BA3b_union") & s1_manifest$variant == "left", , drop = FALSE]
    write_tsv(
      flexible_volume_models(s1, s1_headline_left, "S1_left"),
      "S1_Juelich_left_flexible_lesion_volume_models_four.tsv"
    )

    s1_exclusion_rows <- list()
    for (nm in names(exclusion_candidates)) {
      fc <- resolve_col(s1, exclusion_candidates[[nm]], required = FALSE)
      if (is.na(fc)) next
      flag <- num(s1[[fc]])
      keep <- is.na(flag) | flag == 0
      rr <- apply_s1_fdr(fit_manifest(s1[keep, , drop = FALSE], s1_manifest))
      rr$sensitivity <- nm
      rr$excluded_n <- sum(!keep)
      s1_exclusion_rows[[length(s1_exclusion_rows) + 1L]] <- rr
    }
    write_tsv(bind_rows_base(s1_exclusion_rows), "S1_Juelich_sample_exclusion_sensitivities_four.tsv")
  }

  branch_defs_s1 <- defs[defs$branch %in% c("BranchA", "BranchC") & defs$variant == "left", , drop = FALSE]
  branch_defs_s1 <- branch_defs_s1[grepl("mean_fisher_z|conditional_nonzero_mean", branch_defs_s1$predictor), , drop = FALSE]
  branch_defs_s1 <- branch_defs_s1[!duplicated(paste(branch_defs_s1$roi, branch_defs_s1$branch, branch_defs_s1$modality)), , drop = FALSE]
  if (nrow(branch_defs_s1)) {
    man <- expand_predictors_outcomes(branch_defs_s1, family_prefix = "S1_branch_sensitivity_four", covars = s1_cov$base)
    write_tsv(apply_s1_fdr(fit_manifest(s1, man)), "S1_Juelich_branch_sensitivities_four.tsv")
  }

  s1_alt_defs <- defs[defs$modality == "BCB" & defs$branch == "BranchB" & defs$variant == "left" & grepl("mean_all|extent_pct_nonzero|integrated_probability_sum|p95_all|maximum|conditional_nonzero_mean", defs$predictor), , drop = FALSE]
  s1_alt_rows <- list()
  for (i in seq_len(nrow(s1_alt_defs))) {
    d <- s1_alt_defs[i, ]
    metric <- sub(".*__(conditional_nonzero_mean|mean_all|extent_pct_nonzero|integrated_probability_sum|p95_all|maximum)$", "\\1", d$predictor)
    for (j in seq_len(nrow(OUTCOME_META))) {
      o <- OUTCOME_META[j, ]
      r <- fit_model(s1, o$outcome, d$predictor, s1_cov$base, paste("S1_alt", d$roi, metric, o$outcome_label, sep = "__"), o$binary)
      r$roi <- d$roi
      r$metric <- metric
      r$outcome_label <- o$outcome_label
      s1_alt_rows[[length(s1_alt_rows) + 1L]] <- r
    }
  }
  s1_alt <- bind_rows_base(s1_alt_rows)
  if (nrow(s1_alt)) {
    s1_alt$q_four_outcomes_within_roi_metric <- bh_within(num(s1_alt$p_value), paste(s1_alt$roi, s1_alt$metric, sep = "__"))
    s1_alt$q_metrics_within_roi_outcome <- bh_within(num(s1_alt$p_value), paste(s1_alt$roi, s1_alt$outcome_label, sep = "__"))
    s1_alt$q_all_S1_alt_models <- p.adjust(num(s1_alt$p_value), "BH")
  }
  write_tsv(s1_alt, "S1_Juelich_alternative_BCB_estimands_four.tsv")

  side_outcomes <- c("QST_PC2_side_difference", "QST_PC1_side_difference", "QST_control_PC2_projected", "QST_PC2_control_projected", "QST_control_PC1_projected")
  side_outcomes <- side_outcomes[side_outcomes %in% names(s1)]
  side_rows <- list()
  headline_preds <- unique(main_defs$predictor[main_defs$roi %in% c("BA1", "BA3b") & main_defs$variant == "left"])
  for (pred in headline_preds) for (o in side_outcomes) {
    side_rows[[length(side_rows) + 1L]] <- fit_model(s1, o, pred, s1_cov$base, paste("S1_side", pred, o, sep = "__"), FALSE)
  }
  write_tsv(bind_rows_base(side_rows), "S1_Juelich_PC2_PC1_mirror_side_sensitivities.tsv")
}

morel_manifest <- data.frame()
morel_full <- data.frame()
morel_binary_manifest <- data.frame()
morel_binary_full <- data.frame()
morel_pct_predictor <- NA_character_
morel_any_predictor <- NA_character_

if (!is.null(morel)) {
  raw_pct <- "percent_roi_damaged.whole_thalamus"
  existing_z_pct <- "z_percent_roi_damaged.whole_thalamus"
  any_col <- "any_overlap.whole_thalamus"

  if (existing_z_pct %in% names(morel)) {
    morel_pct_predictor <- existing_z_pct
  } else if (raw_pct %in% names(morel)) {
    morel$z_percent_roi_damaged_whole_thalamus_four <- safe_z(morel[[raw_pct]])
    morel_pct_predictor <- "z_percent_roi_damaged_whole_thalamus_four"
  }

  if (any_col %in% names(morel)) {
    morel[[any_col]] <- as.integer(num(morel[[any_col]]) > 0)
    morel_any_predictor <- any_col
  }

  if (!is.na(morel_pct_predictor)) {
    pct_def <- data.frame(
      roi = "whole_thalamus",
      variant = "continuous_percentage_damage",
      modality = "direct_anatomy",
      branch = "Morel_1mm",
      predictor = morel_pct_predictor,
      stringsAsFactors = FALSE
    )
    morel_manifest <- expand_predictors_outcomes(
      pct_def,
      family_prefix = "Morel_whole_thalamus_percentage_four",
      covars = morel_cov$base
    )
    morel_full <- fit_manifest(morel, morel_manifest)
    morel_full$q_Morel_percentage_four <- p.adjust(num(morel_full$p_value), "BH")
    morel_full$Holm_Morel_percentage_four <- p.adjust(num(morel_full$p_value), "holm")
    write_tsv(morel_manifest, "Morel_whole_thalamus_percentage_four_manifest.tsv")
    write_tsv(morel_full, "Morel_whole_thalamus_percentage_four_full.tsv")

    morel_sens_rows <- list()
    for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
      cv <- morel_cov[[nm]]
      if (length(cv) <= length(morel_cov$base)) next
      man <- expand_predictors_outcomes(
        pct_def,
        family_prefix = "Morel_whole_thalamus_percentage_four",
        covars = cv,
        adjustment = nm
      )
      rr <- fit_manifest(morel, man)
      rr$q_Morel_percentage_four <- p.adjust(num(rr$p_value), "BH")
      rr$Holm_Morel_percentage_four <- p.adjust(num(rr$p_value), "holm")
      morel_sens_rows[[length(morel_sens_rows) + 1L]] <- rr
    }
    write_tsv(bind_rows_base(morel_sens_rows), "Morel_whole_thalamus_age_sex_chronicity_sensitivities_four.tsv")

    morel_exclusion_rows <- list()
    exclusion_specs <- list(
      exclude_top5 = c("sensitivity_exclude_large_lesion_top5", "large_lesion_top5_flag"),
      exclude_top10 = c("sensitivity_exclude_large_lesion_top10", "large_lesion_top10_flag")
    )
    for (nm in names(exclusion_specs)) {
      spec <- exclusion_specs[[nm]]
      fc <- resolve_col(morel, spec, required = FALSE)
      if (!is.na(fc)) {
        keep <- is.na(num(morel[[fc]])) | num(morel[[fc]]) == 0
      } else {
        main_fc <- resolve_col(main, spec, required = FALSE)
        if (is.na(main_fc)) next
        excluded_ids <- main$participant_id[is.finite(num(main[[main_fc]])) & num(main[[main_fc]]) != 0]
        keep <- !morel$participant_id %in% excluded_ids
      }
      rr <- fit_manifest(morel[keep, , drop = FALSE], morel_manifest)
      rr$q_Morel_percentage_four <- p.adjust(num(rr$p_value), "BH")
      rr$sensitivity <- nm
      rr$excluded_n <- sum(!keep)
      morel_exclusion_rows[[length(morel_exclusion_rows) + 1L]] <- rr
    }
    write_tsv(bind_rows_base(morel_exclusion_rows), "Morel_whole_thalamus_large_lesion_exclusions_four.tsv")
  }

  if (!is.na(morel_any_predictor)) {
    morel_binary_manifest <- data.frame(
      model_id = "Morel_whole_thalamus_any_overlap__CPSP",
      analysis_family = "Morel_binary_entry",
      roi = "whole_thalamus",
      variant = "any_overlap",
      modality = "direct_anatomy",
      branch = "Morel_1mm",
      outcome = "CPSP_status_binary",
      outcome_label = "CPSP_status",
      domain = "clinical",
      binary = TRUE,
      predictor = morel_any_predictor,
      adjustment = "minimal",
      stringsAsFactors = FALSE
    )
    morel_binary_manifest$covariates <- I(list(morel_cov$base))
    morel_binary_full <- fit_manifest(morel, morel_binary_manifest)
    write_tsv(morel_binary_full, "Morel_whole_thalamus_binary_entry_CPSP.tsv")

    binary_sens_rows <- list()
    for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
      cv <- morel_cov[[nm]]
      if (length(cv) <= length(morel_cov$base)) next
      man <- morel_binary_manifest
      man$model_id <- paste0(man$model_id, "__", nm)
      man$adjustment <- nm
      man$covariates <- I(list(cv))
      binary_sens_rows[[length(binary_sens_rows) + 1L]] <- fit_manifest(morel, man)
    }
    write_tsv(bind_rows_base(binary_sens_rows), "Morel_binary_entry_age_sex_chronicity_sensitivities.tsv")

    binary_exclusion_rows <- list()
    binary_exclusion_specs <- list(
      exclude_top5 = c("sensitivity_exclude_large_lesion_top5", "large_lesion_top5_flag"),
      exclude_top10 = c("sensitivity_exclude_large_lesion_top10", "large_lesion_top10_flag")
    )
    for (nm in names(binary_exclusion_specs)) {
      spec <- binary_exclusion_specs[[nm]]
      fc <- resolve_col(morel, spec, required = FALSE)
      if (!is.na(fc)) {
        keep <- is.na(num(morel[[fc]])) | num(morel[[fc]]) == 0
      } else {
        main_fc <- resolve_col(main, spec, required = FALSE)
        if (is.na(main_fc)) next
        excluded_ids <- main$participant_id[is.finite(num(main[[main_fc]])) & num(main[[main_fc]]) != 0]
        keep <- !morel$participant_id %in% excluded_ids
      }
      rr <- fit_manifest(morel[keep, , drop = FALSE], morel_binary_manifest)
      rr$sensitivity <- nm
      rr$excluded_n <- sum(!keep)
      binary_exclusion_rows[[length(binary_exclusion_rows) + 1L]] <- rr
    }
    write_tsv(bind_rows_base(binary_exclusion_rows), "Morel_binary_entry_large_lesion_exclusions.tsv")
  }

  sparse_targets <- c("VPL", "VPI", "PuA", "posterior_sensory_borderzone")
  sparse_rows <- list()
  sparse_participants <- list()
  for (target in sparse_targets) {
    col <- paste0("any_overlap.", target)
    pct <- paste0("percent_roi_damaged.", target)
    if (!col %in% names(morel)) next
    x <- as.integer(num(morel[[col]]) > 0)
    y <- morel$CPSP_status_binary
    keep <- is.finite(x) & is.finite(y)
    tab <- table(factor(x[keep], levels = c(0, 1)), factor(y[keep], levels = c(0, 1)))
    ft <- tryCatch(fisher.test(tab, conf.int = TRUE), error = function(e) NULL)
    sparse_rows[[length(sparse_rows) + 1L]] <- data.frame(
      target = target,
      source_any_overlap = col,
      source_percentage_damage = ifelse(pct %in% names(morel), pct, NA_character_),
      overlap_n = sum(x[keep] == 1),
      CPSP_overlap = sum(x[keep] == 1 & y[keep] == 1),
      nonCPSP_overlap = sum(x[keep] == 1 & y[keep] == 0),
      exact_OR = if (!is.null(ft)) unname(ft$estimate) else NA,
      exact_CI_low = if (!is.null(ft)) ft$conf.int[1] else NA,
      exact_CI_high = if (!is.null(ft)) ft$conf.int[2] else NA,
      p_value = if (!is.null(ft)) ft$p.value else NA,
      status = ifelse(sum(x[keep] == 1) < 10, "SPARSE_DESCRIPTIVE", "EXACT_SENSITIVITY"),
      stringsAsFactors = FALSE
    )
    idx <- which(keep & x == 1)
    if (length(idx)) {
      sparse_participants[[length(sparse_participants) + 1L]] <- data.frame(
        target = target,
        participant_id = morel$participant_id[idx],
        CPSP_status_binary = y[idx],
        NPSI_total = morel$NPSI_total_four[idx],
        QST_PC2 = morel$QST_PC2_four[idx],
        QST_PC1 = morel$QST_PC1_four[idx],
        percent_roi_damaged = if (pct %in% names(morel)) num(morel[[pct]][idx]) else NA,
        stringsAsFactors = FALSE
      )
    }
  }
  sparse <- bind_rows_base(sparse_rows)
  if (nrow(sparse)) sparse$q_BH_sparse_targets <- p.adjust(num(sparse$p_value), "BH")
  write_tsv(sparse, "Morel_sparse_targets_CPSP_exact.tsv")
  write_tsv(bind_rows_base(sparse_participants), "Morel_sparse_target_overlap_participants.tsv")

  morph_cols <- grep(
    "whole.*thalam.*(eros|dilat|original).*(any|overlap|entry|binary)|(any|overlap|entry|binary).*whole.*thalam.*(eros|dilat|original)",
    names(morel), ignore.case = TRUE, value = TRUE
  )
  morph_rows <- list()
  for (col in unique(morph_cols)) {
    x <- num(morel[[col]])
    if (all(is.na(x)) || length(unique(x[is.finite(x)])) < 2L) next
    tmp <- paste0("tmp_morph_", length(morph_rows) + 1L)
    morel[[tmp]] <- as.integer(x > 0)
    r <- fit_model(
      morel, "CPSP_status_binary", tmp, morel_cov$base,
      paste("Morel_morph", col, sep = "__"), TRUE
    )
    r$source_column <- col
    morph_rows[[length(morph_rows) + 1L]] <- r
  }
  morph <- bind_rows_base(morph_rows)
  if (nrow(morph)) morph$q_BH_morphology <- p.adjust(num(morph$p_value), "BH")
  write_tsv(morph, "Morel_binary_entry_morphology_sensitivities.tsv")
}

if (RUN_FULL_LOO) {
  primary_loo <- loo_manifest(main, primary_manifest, apply_primary_fdr)
  write_tsv(primary_loo, "primary_BA3a_BA3c_four_LOO_long.tsv")
  write_tsv(summarise_loo(primary_full, primary_loo, c("q_within_modality_4", "q_combined_modalities_8")), "primary_BA3a_BA3c_four_LOO_summary.tsv")

  structured_loo <- loo_manifest(main, structured_manifest, apply_structured_fdr)
  write_tsv(structured_loo, "structured_secondary_24_LOO_long.tsv")
  write_tsv(summarise_loo(structured_full, structured_loo, c("q_domain_modality_6", "q_all4_modality_12", "q_domain_combined_modalities_12", "q_all4_combined_modalities_24")), "structured_secondary_24_LOO_summary.tsv")

  if (nrow(localisation_manifest)) {
    localisation_loo <- loo_manifest(main, localisation_manifest, apply_localisation_fdr)
    write_tsv(localisation_loo, "HCPex_localisation_four_LOO_long.tsv")
    write_tsv(
      summarise_loo(localisation_full, localisation_loo, c("q_localisation_all_four", "q_localisation_by_modality")),
      "HCPex_localisation_four_LOO_summary.tsv"
    )
  }

  if (!is.null(s1) && nrow(s1_manifest)) {
    s1_loo <- loo_manifest(s1, s1_manifest, apply_s1_fdr)
    write_tsv(s1_loo, "S1_Juelich_four_LOO_long.tsv")
    write_tsv(summarise_loo(s1_full, s1_loo, c("q_within_roi_variant_modality_4", "q_combined_modalities_within_roi_variant_8")), "S1_Juelich_four_LOO_summary.tsv")
  }

  if (!is.null(morel) && nrow(morel_manifest)) {
    morel_loo <- loo_manifest(morel, morel_manifest, apply_morel_percentage_fdr)
    write_tsv(morel_loo, "Morel_whole_thalamus_percentage_four_LOO_long.tsv")
    write_tsv(
      summarise_loo(morel_full, morel_loo, c("q_Morel_percentage_four")),
      "Morel_whole_thalamus_percentage_four_LOO_summary.tsv"
    )
  }
  if (!is.null(morel) && nrow(morel_binary_manifest)) {
    morel_binary_loo <- loo_manifest(morel, morel_binary_manifest, NULL)
    write_tsv(morel_binary_loo, "Morel_binary_entry_CPSP_LOO_long.tsv")
    write_tsv(summarise_loo(morel_binary_full, morel_binary_loo), "Morel_binary_entry_CPSP_LOO_summary.tsv")
  }
}

primary_influence <- influence_for_manifest(main, primary_manifest, "primary")
structured_influence <- influence_for_manifest(main, structured_manifest, "structured")
localisation_influence <- if (nrow(localisation_manifest)) influence_for_manifest(main, localisation_manifest, "HCPex_localisation") else data.frame()
s1_influence <- if (!is.null(s1) && nrow(s1_manifest)) influence_for_manifest(s1, s1_manifest, "S1") else data.frame()
morel_influence <- if (!is.null(morel) && nrow(morel_manifest)) influence_for_manifest(morel, morel_manifest, "Morel") else data.frame()
write_tsv(bind_rows_base(list(primary_influence, structured_influence, localisation_influence, s1_influence, morel_influence)), "four_outcome_influence_diagnostics_long.tsv")

primary_robust <- robust_for_manifest(main, primary_manifest, "primary")
structured_robust <- robust_for_manifest(main, structured_manifest, "structured")
localisation_robust <- if (nrow(localisation_manifest)) robust_for_manifest(main, localisation_manifest, "HCPex_localisation") else data.frame()
s1_robust <- if (!is.null(s1) && nrow(s1_manifest)) robust_for_manifest(s1, s1_manifest, "S1") else data.frame()
morel_robust <- if (!is.null(morel) && nrow(morel_manifest)) robust_for_manifest(morel, morel_manifest, "Morel") else data.frame()
write_tsv(bind_rows_base(list(primary_robust, structured_robust, localisation_robust, s1_robust, morel_robust)), "four_outcome_Huber_robust_regression.tsv")

if (RUN_BOOTSTRAP) {
  selected <- bind_rows_base(list(
    primary_manifest,
    structured_manifest[structured_manifest$roi == "operculo_insular_S2_composite", , drop = FALSE]
  ))
  if (!is.null(s1) && nrow(s1_manifest)) {
    selected_s1 <- s1_manifest[s1_manifest$roi %in% c("BA1", "BA3b", "BA1_BA3b_union") & s1_manifest$variant == "left", , drop = FALSE]
  } else selected_s1 <- data.frame()

  boot_rows <- list()
  if (nrow(selected)) {
    boot_rows <- c(boot_rows, parallel_lapply(seq_len(nrow(selected)), function(i) {
      m <- selected[i, , drop = FALSE]
      bootstrap_model(main, m$outcome, m$predictor, selected$covariates[[i]], m$binary, BOOT_REPS, m$model_id)
    }))
  }
  if (nrow(selected_s1)) {
    boot_rows <- c(boot_rows, parallel_lapply(seq_len(nrow(selected_s1)), function(i) {
      m <- selected_s1[i, , drop = FALSE]
      bootstrap_model(s1, m$outcome, m$predictor, selected_s1$covariates[[i]], m$binary, BOOT_REPS, m$model_id)
    }))
  }
  if (!is.null(morel) && nrow(morel_manifest)) {
    boot_rows <- c(boot_rows, parallel_lapply(seq_len(nrow(morel_manifest)), function(i) {
      m <- morel_manifest[i, , drop = FALSE]
      bootstrap_model(morel, m$outcome, m$predictor, morel_manifest$covariates[[i]], m$binary, BOOT_REPS, m$model_id)
    }))
  }
  if (!is.null(morel) && nrow(morel_binary_manifest)) {
    m <- morel_binary_manifest[1, , drop = FALSE]
    boot_rows[[length(boot_rows) + 1L]] <- bootstrap_model(
      morel, m$outcome, m$predictor, morel_binary_manifest$covariates[[1]],
      TRUE, BOOT_REPS, m$model_id
    )
  }
  write_tsv(bind_rows_base(boot_rows), "four_outcome_headline_bootstrap.tsv")
}

diag_manifest <- bind_rows_base(list(
  primary_manifest[primary_manifest$binary, , drop = FALSE],
  structured_manifest[structured_manifest$binary, , drop = FALSE]
))
if (!is.null(s1) && nrow(s1_manifest)) diag_manifest <- bind_rows_base(list(diag_manifest, s1_manifest[s1_manifest$binary, , drop = FALSE]))
if (!is.null(morel) && nrow(morel_binary_manifest)) diag_manifest <- bind_rows_base(list(diag_manifest, morel_binary_manifest))

standard_binary_rows <- list()
firth_binary_rows <- list()
logistic_diag_rows <- list()
logistic_probability_rows <- list()
for (i in seq_len(nrow(diag_manifest))) {
  m <- diag_manifest[i, , drop = FALSE]
  dat_use <- if (grepl("^S1_", m$analysis_family)) s1 else if (grepl("^Morel", m$analysis_family)) morel else main
  covs <- diag_manifest$covariates[[i]]
  d <- complete_model_data(dat_use, c("participant_id", m$outcome, m$predictor, covs))
  form <- as.formula(formula_string(m$outcome, c(m$predictor, covs)))
  std <- standard_logistic_fit(form, d, m$predictor, m$model_id)
  fir <- firth_fit(form, d, m$predictor, m$model_id)
  standard_binary_rows[[length(standard_binary_rows) + 1L]] <- std
  firth_binary_rows[[length(firth_binary_rows) + 1L]] <- fir

  g <- tryCatch(glm(form, data = d, family = binomial()), error = function(e) NULL)
  if (!is.null(g)) {
    prb <- fitted(g)
    co <- coef(g)
    logistic_diag_rows[[length(logistic_diag_rows) + 1L]] <- data.frame(
      model_id = m$model_id,
      n = nrow(d),
      events = sum(num(d[[m$outcome]]) == 1),
      parameters = length(co),
      events_per_parameter = sum(num(d[[m$outcome]]) == 1) / length(co),
      converged = isTRUE(g$converged),
      rank_deficient = g$rank < length(co),
      maximum_absolute_coefficient = max(abs(co), na.rm = TRUE),
      maximum_finite_odds_ratio = max(exp(pmin(abs(co), 700)), na.rm = TRUE),
      predicted_probability_min = min(prb, na.rm = TRUE),
      predicted_probability_max = max(prb, na.rm = TRUE),
      extreme_probability_n = sum(prb < 1e-6 | prb > 1 - 1e-6, na.rm = TRUE),
      instability_flag = any(abs(co) > 10, na.rm = TRUE) || any(prb < 1e-6 | prb > 1 - 1e-6, na.rm = TRUE) || !isTRUE(g$converged),
      stringsAsFactors = FALSE
    )
    logistic_probability_rows[[length(logistic_probability_rows) + 1L]] <- data.frame(
      model_id = m$model_id,
      participant_id = d$participant_id,
      observed_CPSP = num(d[[m$outcome]]),
      predicted_probability_standard_logistic = prb,
      stringsAsFactors = FALSE
    )
  }
}
std_binary <- bind_rows_base(standard_binary_rows)
fir_binary <- bind_rows_base(firth_binary_rows)
write_tsv(std_binary, "four_outcome_standard_logistic_companions.tsv")
write_tsv(fir_binary, "four_outcome_Firth_principal_CPSP_models.tsv")
write_tsv(bind_rows_base(logistic_diag_rows), "four_outcome_logistic_stability_diagnostics.tsv")
write_tsv(bind_rows_base(logistic_probability_rows), "four_outcome_standard_logistic_predicted_probabilities.tsv")

binary_compare <- merge(
  std_binary[, intersect(c("model_id", "estimate", "ci_low", "ci_high", "odds_ratio", "OR_ci_low", "OR_ci_high", "p_value", "converged", "status"), names(std_binary)), drop = FALSE],
  fir_binary[, intersect(c("model_id", "estimate", "ci_low", "ci_high", "odds_ratio", "OR_ci_low", "OR_ci_high", "p_value", "converged", "status"), names(fir_binary)), drop = FALSE],
  by = "model_id", all = TRUE, suffixes = c("_standard", "_Firth")
)
write_tsv(binary_compare, "four_outcome_standard_vs_Firth_comparison.tsv")

diag_cv_rows <- parallel_lapply(seq_len(nrow(diag_manifest)), function(i) {
  m <- diag_manifest[i, , drop = FALSE]
  dat_use <- if (grepl("^S1_", m$analysis_family)) s1 else if (grepl("^Morel", m$analysis_family)) morel else main
  loocv_binary(dat_use, m$outcome, m$predictor, diag_manifest$covariates[[i]], m$model_id)
})
write_tsv(bind_rows_base(diag_cv_rows), "four_outcome_CPSP_LOOCV_AUC_Brier_calibration.tsv")

existing_perm_paths <- Sys.getenv("EXISTING_PERMUTATION_TABLES", "")
perm_recalc <- list()
if (nzchar(existing_perm_paths)) {
  paths <- strsplit(existing_perm_paths, ":", fixed = TRUE)[[1]]
  paths <- paths[file.exists(paths)]
  retained_patterns <- c("CPSP", "NPSI_total", "PC2", "PC1")
  excluded_patterns <- c("NPSI_evoked", "temporal", "WUR", "DN4")
  for (path in paths) {
    tab <- read_any(path)
    text_cols <- names(tab)[vapply(tab, function(x) is.character(x) || is.factor(x), logical(1))]
    key <- if (length(text_cols)) apply(tab[, text_cols, drop = FALSE], 1, paste, collapse = " | ") else rep("", nrow(tab))
    keep <- Reduce(`|`, lapply(retained_patterns, function(p) grepl(p, key, ignore.case = TRUE))) & !Reduce(`|`, lapply(excluded_patterns, function(p) grepl(p, key, ignore.case = TRUE)))
    z <- tab[keep, , drop = FALSE]
    pcol <- resolve_col(z, c("permutation_p_two_sided", "permutation_p", "p_value", "p"), required = FALSE)
    if (!is.na(pcol) && nrow(z)) {
      pp <- num(z[[pcol]])
      roi_col <- resolve_col(z, c("roi", "target", "region"), required = FALSE)
      variant_col <- resolve_col(z, c("variant", "laterality", "hemisphere"), required = FALSE)
      modality_col <- resolve_col(z, c("modality"), required = FALSE)
      branch_col <- resolve_col(z, c("branch"), required = FALSE)
      available <- c(roi_col, variant_col, modality_col, branch_col)
      available <- available[!is.na(available)]
      if (length(available)) {
        grp <- apply(z[, available, drop = FALSE], 1, paste, collapse = "__")
        z$q_BH_within_detected_model_family <- bh_within(pp, grp)
        z$Holm_within_detected_model_family <- holm_within(pp, grp)
      }
      z$q_BH_global_retained_rows_transparency <- p.adjust(pp, "BH")
      z$Holm_global_retained_rows_transparency <- p.adjust(pp, "holm")
    }
    z$source_table <- path
    perm_recalc[[length(perm_recalc) + 1L]] <- z
  }
}
write_tsv(bind_rows_base(perm_recalc), "existing_ROI_permutation_pvalues_recorrected_four_outcomes.tsv")

write_tsv(main, "prepared_main.tsv", "audit")
if (!is.null(s1)) write_tsv(s1, "prepared_S1.tsv", "audit")
if (!is.null(morel)) write_tsv(morel, "prepared_Morel.tsv", "audit")

pc2_provenance <- data.frame(
  dataset = c("main", if (!is.null(s1)) "S1" else character(), if (!is.null(morel)) "Morel" else character()),
  source = c(attr(main, "pc2_source"), if (!is.null(s1)) attr(s1, "pc2_source") else character(), if (!is.null(morel)) attr(morel, "pc2_source") else character()),
  n_finite = c(sum(is.finite(num(main$QST_PC2_four))), if (!is.null(s1)) sum(is.finite(num(s1$QST_PC2_four))) else integer(), if (!is.null(morel)) sum(is.finite(num(morel$QST_PC2_four))) else integer()),
  n_total = c(nrow(main), if (!is.null(s1)) nrow(s1) else integer(), if (!is.null(morel)) nrow(morel) else integer()),
  recomputed = FALSE,
  sign_reversed = FALSE,
  stringsAsFactors = FALSE
)
write_tsv(pc2_provenance, "QST_PC2_PROVENANCE.tsv", "audit")

write_tsv(data.frame(
  component = "QST_PC2",
  orientation = "as stored in the participant-level source dataset",
  intended_interpretation = "Higher saved PC2: relatively greater mechanical/vibration gain with lower thermal-pain and wind-up responses",
  numeric_loadings_in_bundle = FALSE,
  warning = "Confirm against the original PCA loading table before manuscript interpretation; this workflow does not recompute or reorient PCA scores.",
  stringsAsFactors = FALSE
), "QST_PC2_DIRECTION_REFERENCE.tsv", "audit")

flatten_manifest <- function(manifest, dataset, permutation_family = NULL,
                             resample_bootstrap = TRUE,
                             resample_permutation = TRUE,
                             resample_loo = TRUE) {
  if (is.null(manifest) || !is.data.frame(manifest) || !nrow(manifest)) return(data.frame())
  z <- manifest
  z$dataset <- dataset
  if (!"analysis_family" %in% names(z)) z$analysis_family <- "unspecified"
  if (!"roi" %in% names(z)) z$roi <- "unspecified"
  if (!"variant" %in% names(z)) z$variant <- "unspecified"
  if (!"modality" %in% names(z)) z$modality <- "unspecified"
  if (!"branch" %in% names(z)) z$branch <- "unspecified"
  if (!"adjustment" %in% names(z)) z$adjustment <- "minimal"
  z$covariates_flat <- vapply(z$covariates, function(x) paste(as.character(unlist(x)), collapse = "+"), character(1))
  if (is.null(permutation_family)) {
    z$permutation_family <- paste(z$analysis_family, z$modality, z$branch, z$variant, z$adjustment, sep = "__")
  } else {
    z$permutation_family <- permutation_family
  }
  z$resample_bootstrap <- resample_bootstrap
  z$resample_permutation <- resample_permutation & !as.logical(z$binary)
  z$resample_loo <- resample_loo
  keep <- c("dataset", "model_id", "analysis_family", "permutation_family", "roi", "variant", "modality", "branch", "outcome", "outcome_label", "domain", "binary", "predictor", "covariates_flat", "adjustment", "resample_bootstrap", "resample_permutation", "resample_loo")
  z[, keep, drop = FALSE]
}

adjusted_manifests <- list()
append_adjusted <- function(base_predictors, family, covset, dataset) {
  rows <- list()
  for (nm in c("age_sex", "chronic", "age_sex_chronic")) {
    cv <- covset[[nm]]
    if (length(cv) <= length(covset$base)) next
    rows[[length(rows) + 1L]] <- flatten_manifest(
      expand_predictors_outcomes(base_predictors, family_prefix = family, covars = cv, adjustment = nm),
      dataset
    )
  }
  bind_rows_base(rows)
}

registry_parts <- list(
  flatten_manifest(primary_manifest, "main"),
  flatten_manifest(primary_branch_manifest, "main"),
  flatten_manifest(structured_manifest, "main"),
  flatten_manifest(structured_branch_manifest, "main"),
  flatten_manifest(localisation_manifest, "main"),
  append_adjusted(primary_predictors, "primary_BA3a_BA3c_heat_only_four", main_cov, "main"),
  append_adjusted(structured_defs, "structured_secondary_four", main_cov, "main")
)
if (exists("loc_defs") && nrow(loc_defs)) {
  registry_parts[[length(registry_parts) + 1L]] <- append_adjusted(loc_defs, "HCPex_localisation_four", main_cov, "main")
}
if (!is.null(s1) && nrow(s1_manifest)) {
  registry_parts[[length(registry_parts) + 1L]] <- flatten_manifest(s1_manifest, "S1")
  if (exists("main_defs") && nrow(main_defs)) registry_parts[[length(registry_parts) + 1L]] <- append_adjusted(main_defs, "S1_Juelich_four", s1_cov, "S1")
}
if (!is.null(morel) && nrow(morel_manifest)) {
  registry_parts[[length(registry_parts) + 1L]] <- flatten_manifest(morel_manifest, "Morel")
  if (exists("pct_def") && nrow(pct_def)) registry_parts[[length(registry_parts) + 1L]] <- append_adjusted(pct_def, "Morel_whole_thalamus_percentage_four", morel_cov, "Morel")
}
if (!is.null(morel) && nrow(morel_binary_manifest)) {
  registry_parts[[length(registry_parts) + 1L]] <- flatten_manifest(morel_binary_manifest, "Morel")
}
registry <- bind_rows_base(registry_parts)
if (nrow(registry)) {
  registry <- registry[!duplicated(paste(registry$dataset, registry$model_id, sep = "__")), , drop = FALSE]
  registry$registry_index <- seq_len(nrow(registry))
  registry <- registry[, c("registry_index", setdiff(names(registry), "registry_index")), drop = FALSE]
}
write_tsv(registry, "PARALLEL_MODEL_REGISTRY.tsv", "audit")

write_tsv(data.frame(
  analysis = c(
    "sample-exclusion sensitivities",
    "nonlinear lesion-volume and imaging-by-volume models",
    "technical-QC covariate sensitivities",
    "alternative BCB estimands",
    "joint extent-and-intensity models",
    "exact sparse-target tests and morphology perturbations",
    "mirror/reference-side exact PC2/PC1 sensitivities",
    "internal prediction diagnostics"
  ),
  array_resampled = FALSE,
  reason = c(
    "requires analysis-specific participant subsets",
    "requires term-specific nonlinear formula construction",
    "requires analysis-specific QC columns",
    "exploratory estimand family retained as deterministic sensitivity",
    "multi-predictor model requires a separate joint-term resampling design",
    "exact/descriptive sparse analysis is not compatible with generic resampling worker",
    "only run when exact saved side-score columns exist",
    "already implemented as deterministic/LOOCV diagnostics"
  ),
  stringsAsFactors = FALSE
), "PARALLEL_RESAMPLING_SCOPE.tsv", "audit")

required_outputs <- c(
  "tables/primary_BA3a_BA3c_four_full.tsv",
  "tables/primary_BA3a_BA3c_branch_sensitivities_four.tsv",
  "tables/primary_BA3a_BA3c_branch_sensitivities_manifest.tsv",
  "tables/structured_secondary_24_full.tsv",
  "audit/prepared_main.tsv",
  "audit/PARALLEL_MODEL_REGISTRY.tsv",
  "audit/QST_PC2_PROVENANCE.tsv"
)
if (!is.null(s1)) required_outputs <- c(required_outputs, "tables/S1_Juelich_four_full.tsv", "audit/prepared_S1.tsv")
if (!is.null(morel)) required_outputs <- c(required_outputs, "audit/prepared_Morel.tsv")
completion <- data.frame(
  required_output = required_outputs,
  exists = file.exists(file.path(OUT, required_outputs)),
  nonempty = file.info(file.path(OUT, required_outputs))$size > 0,
  stringsAsFactors = FALSE
)
completion$pass <- completion$exists & completion$nonempty
write_tsv(completion, "PREP_SENSITIVITY_COMPLETION_AUDIT.tsv", "audit")
write_tsv(completion[!completion$pass, , drop = FALSE], "PREP_REQUIRED_SENSITIVITY_FAILURES.tsv", "audit")

active_manifest_text <- paste(c(
  if (nrow(primary_manifest)) primary_manifest$outcome else character(),
  if (nrow(primary_branch_manifest)) primary_branch_manifest$outcome else character(),
  if (nrow(structured_manifest)) structured_manifest$outcome else character(),
  if (nrow(localisation_manifest)) localisation_manifest$outcome else character(),
  if (nrow(s1_manifest)) s1_manifest$outcome else character(),
  if (nrow(morel_manifest)) morel_manifest$outcome else character()
), collapse = " | ")
excluded_active <- grepl("thermal|NPSI_evoked|temporal|WUR|DN4", active_manifest_text, ignore.case = TRUE)

qc <- data.frame(
  check = c(
    "main_participants", "main_CPSP_events", "PC2_complete_main", "PC1_complete_main",
    "primary_manifest_rows", "primary_full_rows",
    "heat_only_branch_manifest_rows", "heat_only_branch_full_rows",
    "structured_manifest_rows", "structured_full_rows",
    "parallel_registry_nonempty", "continuous_permutation_models_nonempty",
    "no_excluded_outcomes_in_active_manifests", "required_prep_outputs_complete",
    "heat_only_all_branches_flag_complete",
    "randomise_rerun_requested", "PALM_rerun_requested"
  ),
  observed = c(
    nrow(main), sum(main$CPSP_status_binary == 1, na.rm = TRUE),
    sum(is.finite(num(main$QST_PC2_four))), sum(is.finite(num(main$QST_PC1_four))),
    nrow(primary_manifest), nrow(primary_full),
    nrow(primary_branch_manifest), nrow(primary_branch_results),
    nrow(structured_manifest), nrow(structured_full),
    nrow(registry), sum(registry$resample_permutation, na.rm = TRUE),
    as.integer(!excluded_active), as.integer(all(completion$pass)),
    sum(num(main[[heat_flag]]) == 1, na.rm = TRUE),
    0, 0
  ),
  expected = c(63, 24, 63, 63, 8, 8, 12, 12, 24, 24, ">0", ">0", 1, 1, 63, 0, 0),
  stringsAsFactors = FALSE
)
qc$pass <- c(
  qc$observed[1] == 63,
  qc$observed[2] == 24,
  qc$observed[3] == 63,
  qc$observed[4] == 63,
  qc$observed[5] == 8,
  qc$observed[6] == 8,
  qc$observed[7] == 12,
  qc$observed[8] == 12,
  qc$observed[9] == 24,
  qc$observed[10] == 24,
  qc$observed[11] > 0,
  qc$observed[12] > 0,
  qc$observed[13] == 1,
  qc$observed[14] == 1,
  qc$observed[15] == 63,
  TRUE,
  TRUE
)
write_tsv(qc, "FINAL_PC2_PREP_QC.tsv", "audit")

report <- c(
  "# CPSP heat-only all-branch PC2 four-outcome preparation report",
  "",
  paste0("Generated: ", Sys.time()),
  "",
  "## Primary ROI replacement",
  "",
  "The primary Branch B task-defined BA3a/BA3c-transition predictors use the HC19 heat-only ROI. The former heat-or-vibration values are archived in the prepared main dataset but are not analysed.",
  "Matched heat-only Branch A LNM, Branch A BCB and Branch C BCB predictors are included as processing-branch sensitivities.",
  "",
  "## Retained outcomes",
  "",
  "1. CPSP clinical status.",
  "2. NPSI total.",
  "3. Saved participant-level QST PC2.",
  "4. Saved participant-level QST PC1.",
  "",
  "PC2 is read from the existing participant-level dataset. It is not recomputed, rotated or sign-reversed.",
  "NPSI evoked pain, thermal detection, temporal summation/WUR and DN4 are excluded from every newly constructed principal correction family.",
  "",
  "## Parallel stage",
  "",
  "The preparation job exports immutable model-ready datasets and PARALLEL_MODEL_REGISTRY.tsv.",
  "Bootstrap, fresh continuous-outcome Freedman-Lane permutation and leave-one-out analyses are performed by Slurm arrays.",
  "",
  "## Spatial inference",
  "",
  "Randomise and PALM are not rerun because this sensitivity replaces participant-level primary ROI predictors rather than the completed spatial group-contrast inputs.",
  "",
  "## Key audit files",
  "",
  "- audit/HEAT_ONLY_ALL_BRANCHES_REPLACEMENT_AUDIT.tsv",
  "- audit/HEAT_ONLY_ALL_BRANCHES_ARCHIVED_UNION_COLUMNS.tsv",
  "- audit/SOURCE_heat_only_all_branches_QC.tsv",
  "- audit/HEAT_ONLY_PRIMARY_BRANCH_SENSITIVITY_SCOPE.tsv",
  "- audit/QST_PC2_PROVENANCE.tsv",
  "- audit/QST_PC2_DIRECTION_REFERENCE.tsv",
  "- audit/PARALLEL_MODEL_REGISTRY.tsv",
  "- audit/PARALLEL_RESAMPLING_SCOPE.tsv",
  "- audit/PREP_SENSITIVITY_COMPLETION_AUDIT.tsv",
  "- audit/PREP_REQUIRED_SENSITIVITY_FAILURES.tsv",
  "- audit/FINAL_PC2_PREP_QC.tsv"
)
writeLines(report, file.path(OUT, "report", "PC2_PREPARATION_REPORT.md"))

if (!all(qc$pass)) stop("Final PC2 preparation QC failed; inspect audit/FINAL_PC2_PREP_QC.tsv", call. = FALSE)
writeLines("SUCCESS", file.path(OUT, "PC2_PREP_SUCCESS.flag"))
cat("SUCCESS\nOUT=", OUT, "\nREGISTRY_MODELS=", nrow(registry), "\n", sep = "")
if (sink.number(type = "output") > 0L) sink(type = "output")
SOURCE_CPSP_FOUR_OUTCOME_PC2_PREPARE_R

cat > "$WORK/cpsp_pc2_array_worker.R" <<'SOURCE_CPSP_PC2_ARRAY_WORKER_R'
#!/usr/bin/env Rscript
options(stringsAsFactors = FALSE, width = 220, scipen = 8)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop("Usage: cpsp_pc2_array_worker.R <bootstrap|permutation|loo> <analysis_outdir> <shard_id_0based> <shard_count> <bootstrap_reps> <permutation_reps>", call. = FALSE)
}
MODE <- tolower(args[[1]])
OUT <- normalizePath(args[[2]], mustWork = TRUE)
SHARD_ID <- as.integer(args[[3]])
SHARD_COUNT <- as.integer(args[[4]])
BOOT_REPS <- as.integer(args[[5]])
PERM_REPS <- as.integer(args[[6]])
if (!MODE %in% c("bootstrap", "permutation", "loo")) stop("Unknown mode: ", MODE, call. = FALSE)
if (!is.finite(SHARD_ID) || !is.finite(SHARD_COUNT) || SHARD_ID < 0L || SHARD_ID >= SHARD_COUNT) stop("Invalid shard specification", call. = FALSE)

num <- function(x) suppressWarnings(as.numeric(as.character(x)))
as_bool <- function(x) toupper(as.character(x)) %in% c("TRUE", "T", "1", "YES")
read_tsv <- function(path) read.table(path, header = TRUE, sep = "\t", quote = "\"", comment.char = "", check.names = FALSE, fill = TRUE, na.strings = c("", "NA", "NaN"))
write_tsv <- function(x, path) write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
safe_inverse <- function(M, tol = sqrt(.Machine$double.eps)) {
  direct <- tryCatch(solve(M), error = function(e) NULL)
  if (!is.null(direct)) return(direct)
  z <- svd(M)
  cutoff <- max(z$d) * tol
  z$v %*% (ifelse(z$d > cutoff, 1 / z$d, 0) * t(z$u))
}
formula_string <- function(outcome, terms) {
  terms <- unique(terms[!is.na(terms) & nzchar(terms)])
  if (!length(terms)) paste(outcome, "~ 1") else paste(outcome, "~", paste(terms, collapse = " + "))
}
parse_covars <- function(x) {
  if (is.na(x) || !nzchar(x)) return(character())
  z <- strsplit(x, "+", fixed = TRUE)[[1]]
  trimws(z[nzchar(trimws(z))])
}
complete_model_data <- function(dat, vars) {
  vars <- unique(vars[!is.na(vars) & nzchar(vars)])
  miss <- setdiff(vars, names(dat))
  if (length(miss)) stop("Missing variables: ", paste(miss, collapse = ", "), call. = FALSE)
  dat[complete.cases(dat[, vars, drop = FALSE]), , drop = FALSE]
}

hc3_term <- function(form, dat, term) {
  fit <- lm(form, data = dat)
  X <- model.matrix(fit)
  idx <- match(term, colnames(X))
  if (is.na(idx)) stop("Target term not estimable: ", term, call. = FALSE)
  e <- residuals(fit); h <- hatvalues(fit)
  inv <- safe_inverse(crossprod(X))
  omega <- e^2 / pmax((1 - h)^2, 1e-12)
  vc <- inv %*% crossprod(X, X * omega) %*% inv
  b <- coef(fit)[idx]; se <- sqrt(pmax(diag(vc)[idx], 0)); df <- max(1, nrow(X) - qr(X)$rank)
  data.frame(estimate = b, standard_error = se, statistic = b / se, p_value = 2 * pt(abs(b / se), df, lower.tail = FALSE), status = "OK")
}

penloglik <- function(beta, X, y) {
  eta <- drop(X %*% beta); p <- plogis(pmax(pmin(eta, 35), -35)); W <- pmax(p * (1 - p), 1e-10)
  detI <- determinant(crossprod(X, X * W), logarithm = TRUE)
  if (detI$sign <= 0) return(-Inf)
  sum(y * log(pmax(p, 1e-15)) + (1 - y) * log(pmax(1 - p, 1e-15))) + 0.5 * as.numeric(detI$modulus)
}
firth_matrix <- function(X, y, maxit = 300L, tol = 1e-8) {
  b <- rep(0, ncol(X)); conv <- FALSE; it <- 0L
  for (it in seq_len(maxit)) {
    eta <- drop(X %*% b); p <- plogis(pmax(pmin(eta, 35), -35)); W <- pmax(p * (1 - p), 1e-10)
    I <- crossprod(X, X * W); inv <- safe_inverse(I); h <- rowSums((X %*% inv) * X) * W
    step <- drop(inv %*% crossprod(X, y - p + h * (0.5 - p)))
    old <- penloglik(b, X, y); fac <- 1
    repeat {
      cand <- b + fac * step; ll <- penloglik(cand, X, y)
      if (is.finite(ll) && ll >= old - 1e-10) break
      fac <- fac / 2
      if (fac < 1e-8) break
    }
    nb <- b + fac * step
    if (max(abs(nb - b)) < tol) { b <- nb; conv <- TRUE; break }
    b <- nb
  }
  eta <- drop(X %*% b); p <- plogis(pmax(pmin(eta, 35), -35)); W <- pmax(p * (1 - p), 1e-10)
  list(beta = b, vcov = safe_inverse(crossprod(X, X * W)), converged = conv, iterations = it)
}
firth_term <- function(form, dat, term) {
  mf <- model.frame(form, data = dat, na.action = na.omit); y <- num(model.response(mf)); X <- model.matrix(form, mf)
  if (length(unique(y)) != 2L) stop("Binary outcome has fewer than two classes", call. = FALSE)
  idx <- match(term, colnames(X)); if (is.na(idx)) stop("Target term not estimable: ", term, call. = FALSE)
  fit <- firth_matrix(X, y); se <- sqrt(pmax(diag(fit$vcov), 0)); z <- fit$beta[idx] / se[idx]
  data.frame(estimate = fit$beta[idx], standard_error = se[idx], statistic = z, p_value = 2 * pnorm(abs(z), lower.tail = FALSE), odds_ratio = exp(fit$beta[idx]), converged = fit$converged, status = ifelse(fit$converged, "OK", "NONCONVERGED"))
}

fit_one <- function(dat, outcome, predictor, covars, binary) {
  vars <- unique(c("participant_id", outcome, predictor, covars))
  d <- complete_model_data(dat, vars)
  if (nrow(d) < 15L) stop("Fewer than 15 complete observations", call. = FALSE)
  d[[outcome]] <- num(d[[outcome]]); d[[predictor]] <- num(d[[predictor]])
  if (length(unique(d[[predictor]])) < 2L) stop("Predictor has fewer than two values", call. = FALSE)
  form <- as.formula(formula_string(outcome, c(predictor, covars)))
  res <- if (binary) firth_term(form, d, predictor) else hc3_term(form, d, predictor)
  list(result = res, data = d, formula = form)
}

bootstrap_one <- function(dat, row) {
  outcome <- row$outcome; predictor <- row$predictor; covars <- parse_covars(row$covariates_flat); binary <- as_bool(row$binary)
  full <- fit_one(dat, outcome, predictor, covars, binary); d <- full$data; n <- nrow(d)
  vals <- rep(NA_real_, BOOT_REPS)
  set.seed(2026072400L + as.integer(row$registry_index))
  if (binary) {
    strata <- split(seq_len(n), d[[outcome]])
    for (b in seq_len(BOOT_REPS)) {
      idx <- unlist(lapply(strata, function(ii) sample(ii, length(ii), replace = TRUE)), use.names = FALSE)
      vals[b] <- tryCatch(fit_one(d[idx, , drop = FALSE], outcome, predictor, covars, TRUE)$result$estimate[1], error = function(e) NA_real_)
    }
  } else {
    for (b in seq_len(BOOT_REPS)) {
      idx <- sample.int(n, n, replace = TRUE)
      vals[b] <- tryCatch(fit_one(d[idx, , drop = FALSE], outcome, predictor, covars, FALSE)$result$estimate[1], error = function(e) NA_real_)
    }
  }
  vals <- vals[is.finite(vals)]
  data.frame(
    registry_index = row$registry_index, model_id = row$model_id, dataset = row$dataset,
    outcome = outcome, predictor = predictor, binary = binary, n = n,
    full_estimate = full$result$estimate[1], full_p_value = full$result$p_value[1],
    bootstrap_reps_requested = BOOT_REPS, bootstrap_reps_valid = length(vals),
    bootstrap_low = if (length(vals)) unname(quantile(vals, 0.025)) else NA,
    bootstrap_high = if (length(vals)) unname(quantile(vals, 0.975)) else NA,
    bootstrap_OR_low = if (binary && length(vals)) exp(unname(quantile(vals, 0.025))) else NA,
    bootstrap_OR_high = if (binary && length(vals)) exp(unname(quantile(vals, 0.975))) else NA,
    status = ifelse(length(vals) >= 0.9 * BOOT_REPS, "OK", "LOW_VALID_BOOTSTRAPS"),
    stringsAsFactors = FALSE
  )
}

permutation_one <- function(dat, row) {
  outcome <- row$outcome; predictor <- row$predictor; covars <- parse_covars(row$covariates_flat)
  vars <- unique(c("participant_id", outcome, predictor, covars)); d <- complete_model_data(dat, vars)
  if (nrow(d) < 15L) stop("Fewer than 15 complete observations", call. = FALSE)
  d[[outcome]] <- num(d[[outcome]]); d[[predictor]] <- num(d[[predictor]])
  full_form <- as.formula(formula_string(outcome, c(predictor, covars)))
  reduced_form <- as.formula(formula_string(outcome, covars))
  full <- lm(full_form, data = d); reduced <- lm(reduced_form, data = d)
  idx <- match(predictor, rownames(summary(full)$coefficients)); if (is.na(idx)) stop("Target term not estimable", call. = FALSE)
  obs <- summary(full)$coefficients[idx, "t value"]
  fitted0 <- fitted(reduced); resid0 <- residuals(reduced); tp <- rep(NA_real_, PERM_REPS)
  set.seed(2026072500L + as.integer(row$registry_index))
  for (b in seq_len(PERM_REPS)) {
    d$.perm_y <- fitted0 + sample(resid0, replace = FALSE)
    pf <- update(full_form, .perm_y ~ .)
    z <- tryCatch(lm(pf, data = d), error = function(e) NULL)
    if (!is.null(z)) {
      tab <- summary(z)$coefficients
      if (predictor %in% rownames(tab)) tp[b] <- tab[predictor, "t value"]
    }
  }
  valid <- tp[is.finite(tp)]
  p <- if (length(valid)) (1 + sum(abs(valid) >= abs(obs))) / (1 + length(valid)) else NA_real_
  data.frame(
    registry_index = row$registry_index, model_id = row$model_id, dataset = row$dataset,
    analysis_family = row$analysis_family, permutation_family = row$permutation_family,
    roi = row$roi, variant = row$variant, modality = row$modality, branch = row$branch,
    outcome = outcome, predictor = predictor, n = nrow(d), observed_t = obs,
    permutation_reps_requested = PERM_REPS, permutation_reps_valid = length(valid),
    permutation_p_two_sided = p,
    status = ifelse(length(valid) >= 0.99 * PERM_REPS, "OK", "LOW_VALID_PERMUTATIONS"),
    stringsAsFactors = FALSE
  )
}

loo_one <- function(dat, row) {
  outcome <- row$outcome; predictor <- row$predictor; covars <- parse_covars(row$covariates_flat); binary <- as_bool(row$binary)
  full <- fit_one(dat, outcome, predictor, covars, binary); d <- full$data
  rows <- vector("list", nrow(d))
  for (i in seq_len(nrow(d))) {
    omitted <- d$participant_id[i]
    rr <- tryCatch(fit_one(d[-i, , drop = FALSE], outcome, predictor, covars, binary)$result, error = function(e) NULL)
    rows[[i]] <- if (is.null(rr)) {
      data.frame(registry_index = row$registry_index, model_id = row$model_id, dataset = row$dataset, outcome = outcome, predictor = predictor, binary = binary, omitted_id = omitted, full_estimate = full$result$estimate[1], estimate = NA, p_value = NA, status = "FIT_FAILED", stringsAsFactors = FALSE)
    } else {
      data.frame(registry_index = row$registry_index, model_id = row$model_id, dataset = row$dataset, outcome = outcome, predictor = predictor, binary = binary, omitted_id = omitted, full_estimate = full$result$estimate[1], estimate = rr$estimate[1], p_value = rr$p_value[1], status = rr$status[1], stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}

registry_path <- file.path(OUT, "audit", "PARALLEL_MODEL_REGISTRY.tsv")
if (!file.exists(registry_path)) stop("Registry missing: ", registry_path, call. = FALSE)
registry <- read_tsv(registry_path)
flag_col <- paste0("resample_", MODE)
if (!flag_col %in% names(registry)) stop("Registry flag missing: ", flag_col, call. = FALSE)
registry <- registry[as_bool(registry[[flag_col]]), , drop = FALSE]
registry <- registry[(as.integer(registry$registry_index) - 1L) %% SHARD_COUNT == SHARD_ID, , drop = FALSE]

data_cache <- new.env(parent = emptyenv())
get_dataset <- function(name) {
  key <- as.character(name)
  if (exists(key, envir = data_cache, inherits = FALSE)) return(get(key, envir = data_cache, inherits = FALSE))
  path <- file.path(OUT, "audit", paste0("prepared_", key, ".tsv"))
  if (!file.exists(path)) stop("Prepared dataset missing: ", path, call. = FALSE)
  d <- read_tsv(path); assign(key, d, envir = data_cache); d
}

result_rows <- list()
if (nrow(registry)) {
  for (i in seq_len(nrow(registry))) {
    row <- registry[i, , drop = FALSE]
    dat <- get_dataset(row$dataset)
    rr <- tryCatch(
      switch(MODE,
        bootstrap = bootstrap_one(dat, row),
        permutation = permutation_one(dat, row),
        loo = loo_one(dat, row)
      ),
      error = function(e) data.frame(
        registry_index = row$registry_index, model_id = row$model_id, dataset = row$dataset,
        outcome = row$outcome, predictor = row$predictor, status = paste0("ERROR: ", conditionMessage(e)),
        stringsAsFactors = FALSE
      )
    )
    result_rows[[length(result_rows) + 1L]] <- rr
  }
}
if (length(result_rows)) {
  all_names <- unique(unlist(lapply(result_rows, names), use.names = FALSE))
  result_rows <- lapply(result_rows, function(z) { for (nm in setdiff(all_names, names(z))) z[[nm]] <- NA; z[, all_names, drop = FALSE] })
  results <- do.call(rbind, result_rows)
} else {
  results <- data.frame(registry_index = integer(), model_id = character(), dataset = character(), outcome = character(), predictor = character(), status = character())
}

mode_dir <- file.path(OUT, "parallel", MODE); success_dir <- file.path(mode_dir, "success")
dir.create(success_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(mode_dir, sprintf("shard_%03d.tsv", SHARD_ID))
write_tsv(results, out_path)
writeLines(c(paste0("mode=", MODE), paste0("shard_id=", SHARD_ID), paste0("models_assigned=", nrow(registry)), paste0("rows_written=", nrow(results))), file.path(success_dir, sprintf("shard_%03d.ok", SHARD_ID)))
cat("SUCCESS mode=", MODE, " shard=", SHARD_ID, " models=", nrow(registry), " rows=", nrow(results), "\n", sep = "")
SOURCE_CPSP_PC2_ARRAY_WORKER_R

cat > "$WORK/cpsp_pc2_merge_report.py" <<'SOURCE_CPSP_PC2_MERGE_REPORT_PY'
#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import html
import math
import os
import re
import sys
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

def fail(message: str) -> None:
    raise SystemExit(message)

def as_bool(value: object) -> bool:
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}

def as_float(value: object) -> float:
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan

def fmt_number(value: object, digits: int = 6) -> str:
    x = as_float(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.{digits}g}"

def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = []
        for raw in reader:
            row = {field: (raw.get(field) or "") for field in fields}
            rows.append(row)
    return fields, rows

def union_fields(tables: Iterable[tuple[list[str], list[dict[str, str]]]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for names, _ in tables:
        for name in names:
            if name not in seen:
                seen.add(name)
                fields.append(name)
    return fields

def combine_tables(tables: list[tuple[list[str], list[dict[str, str]]]]) -> tuple[list[str], list[dict[str, str]]]:
    fields = union_fields(tables)
    rows: list[dict[str, str]] = []
    for _, source_rows in tables:
        for source in source_rows:
            rows.append({field: source.get(field, "") for field in fields})
    return fields, rows

def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

def bh_adjust(values: list[object]) -> list[str]:
    out = [""] * len(values)
    finite = [(i, as_float(value)) for i, value in enumerate(values)]
    finite = [(i, value) for i, value in finite if math.isfinite(value)]
    if not finite:
        return out
    ranked = sorted(finite, key=lambda pair: pair[1])
    m = len(ranked)
    adjusted = [1.0] * m
    running = 1.0
    for pos in range(m - 1, -1, -1):
        _, p = ranked[pos]
        rank = pos + 1
        running = min(running, p * m / rank)
        adjusted[pos] = min(1.0, running)
    for (index, _), value in zip(ranked, adjusted):
        out[index] = f"{value:.12g}"
    return out

def holm_adjust(values: list[object]) -> list[str]:
    out = [""] * len(values)
    finite = [(i, as_float(value)) for i, value in enumerate(values)]
    finite = [(i, value) for i, value in finite if math.isfinite(value)]
    if not finite:
        return out
    ranked = sorted(finite, key=lambda pair: pair[1])
    m = len(ranked)
    running = 0.0
    adjusted: list[float] = []
    for pos, (_, p) in enumerate(ranked):
        value = min(1.0, (m - pos) * p)
        running = max(running, value)
        adjusted.append(running)
    for (index, _), value in zip(ranked, adjusted):
        out[index] = f"{value:.12g}"
    return out

def add_group_adjustment(rows: list[dict[str, str]], p_column: str, group_column: str, output_column: str, method: str) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get(group_column, "")].append(index)
    for indices in groups.values():
        values = [rows[index].get(p_column, "") for index in indices]
        adjusted = bh_adjust(values) if method == "BH" else holm_adjust(values)
        for index, value in zip(indices, adjusted):
            rows[index][output_column] = value

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def normalise_status(value: object) -> str:
    return str(value).strip().upper()

def category_for(path: Path) -> str:
    name = path.name.lower()
    if "definition" in name or "provenance" in name or "registry" in name or "scope" in name or "manifest" in name:
        return "Definitions, provenance and model registries"
    if name.startswith("primary_ba3a_ba3c"):
        return "Primary and branch-sensitivity heat-only BA3a/BA3c-transition analyses"
    if name.startswith("structured_secondary"):
        return "Structured secondary ROI analyses"
    if name.startswith("s1_juelich"):
        return "Juelich S1 extensions"
    if name.startswith("morel"):
        return "Morel thalamic analyses"
    if "hcpex" in name or "localisation" in name:
        return "HCPex and anatomical localisation"
    if "bootstrap" in name or "permutation" in name or "loo" in name:
        return "Parallel resampling outputs"
    if any(token in name for token in ("firth", "huber", "influence", "logistic", "prediction", "auc", "brier", "technical", "laterality", "alternative", "joint_extent", "distributional")):
        return "Robustness, diagnostics and alternative estimands"
    if path.parent.name == "audit" or "qc" in name or "failure" in name or "completion" in name:
        return "Audit and quality control"
    if name.startswith("prepared_"):
        return "Prepared model-ready datasets"
    return "Other statistical outputs"

def significance_columns(fields: list[str]) -> tuple[list[str], list[str], list[str]]:
    p_columns: list[str] = []
    adjusted_columns: list[str] = []
    flag_columns: list[str] = []
    for field in fields:
        lower = field.lower()
        if lower.startswith("q_") or "holm" in lower or "fdr" in lower:
            adjusted_columns.append(field)
        elif (
            lower in {"p", "pvalue", "p_value", "p.value"}
            or "p_value" in lower
            or "pvalue" in lower
            or "permutation_p" in lower
            or "exact_p" in lower
            or re.search(r"(^|_)p($|_)", lower)
        ):
            p_columns.append(field)
        if any(token in lower for token in ("interval_excludes_zero", "interval_excludes_one", "significant", "survives")):
            flag_columns.append(field)
    return p_columns, adjusted_columns, flag_columns

def build_headline_index(table_paths: list[Path], out_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows_out: list[dict[str, str]] = []
    union: list[str] = ["source_table", "trigger_type", "trigger_columns"]
    seen = set(union)
    skip = {
        "ALL_PC2_RESULT_ROWS.tsv",
        "COMPREHENSIVE_HEADLINE_INDEX.tsv",
        "OUTPUT_FILE_INVENTORY.tsv",
        "PARALLEL_LOO_LONG_ALL_MODELS.tsv",
    }
    for path in table_paths:
        if path.name in skip:
            continue
        fields, rows = read_tsv(path)
        if not fields or not rows:
            continue
        p_columns, adjusted_columns, flag_columns = significance_columns(fields)
        for source in rows:
            triggers_adjusted = [column for column in adjusted_columns if math.isfinite(as_float(source.get(column))) and as_float(source.get(column)) < 0.05]
            triggers_p = [column for column in p_columns if math.isfinite(as_float(source.get(column))) and as_float(source.get(column)) < 0.05]
            triggers_flag = [column for column in flag_columns if as_bool(source.get(column, ""))]
            if triggers_adjusted:
                trigger_type = "multiplicity_adjusted_below_0.05"
                trigger_columns = triggers_adjusted
            elif triggers_flag:
                trigger_type = "confidence_interval_or_significance_flag"
                trigger_columns = triggers_flag
            elif triggers_p:
                trigger_type = "nominal_p_below_0.05"
                trigger_columns = triggers_p
            else:
                continue
            row = {
                "source_table": str(path.relative_to(out_dir)),
                "trigger_type": trigger_type,
                "trigger_columns": ";".join(trigger_columns),
            }
            row.update(source)
            for field in row:
                if field not in seen:
                    seen.add(field)
                    union.append(field)
            rows_out.append(row)
    return union, rows_out

def html_table(fields: list[str], rows: list[dict[str, str]]) -> str:
    if not fields:
        return "<p><em>Empty file.</em></p>"
    parts = ["<div class='tablewrap'><table><thead><tr>"]
    parts.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if not rows:
        parts.append("<p><em>Header-only table: zero data rows.</em></p>")
    return "".join(parts)

def markdown_tsv_block(fields: list[str], rows: list[dict[str, str]]) -> str:
    if not fields:
        return "_Empty file._\n"
    lines = ["```tsv", "\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")).replace("\n", " ").replace("\r", " ") for field in fields))
    lines.append("```")
    if not rows:
        lines.append("\n_Header-only table: zero data rows._")
    return "\n".join(lines) + "\n"

def write_comprehensive_report(
    out_dir: Path,
    shards: int,
    bootstrap_reps: int,
    permutation_reps: int,
    registry_rows: list[dict[str, str]],
    failures: list[dict[str, str]],
    qc_fields: list[str],
    qc_rows: list[dict[str, str]],
    headline_fields: list[str],
    headline_rows: list[dict[str, str]],
) -> tuple[Path, Path]:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "PC2_COMPREHENSIVE_SINGLE_REPORT.html"
    md_path = report_dir / "PC2_COMPREHENSIVE_SINGLE_REPORT.md"

    table_paths = sorted((out_dir / "tables").glob("*.tsv")) + sorted((out_dir / "audit").glob("*.tsv"))
    category_map: dict[str, list[Path]] = defaultdict(list)
    for path in table_paths:
        category_map[category_for(path)].append(path)
    category_order = [
        "Definitions, provenance and model registries",
        "Audit and quality control",
        "Primary and branch-sensitivity heat-only BA3a/BA3c-transition analyses",
        "Structured secondary ROI analyses",
        "Juelich S1 extensions",
        "Morel thalamic analyses",
        "HCPex and anatomical localisation",
        "Robustness, diagnostics and alternative estimands",
        "Parallel resampling outputs",
        "Prepared model-ready datasets",
        "Other statistical outputs",
    ]

    text_paths = sorted((out_dir / "audit").glob("*.txt"))
    flag_paths = sorted(out_dir.glob("*.flag"))
    plot_paths = sorted((out_dir / "plots").glob("*")) if (out_dir / "plots").is_dir() else []

    registered_bootstrap = sum(as_bool(row.get("resample_bootstrap")) for row in registry_rows)
    registered_permutation = sum(as_bool(row.get("resample_permutation")) for row in registry_rows)
    registered_loo = sum(as_bool(row.get("resample_loo")) for row in registry_rows)
    total_table_rows = 0
    for path in table_paths:
        _, rows = read_tsv(path)
        total_table_rows += len(rows)

    summary_lines = [
        "# CPSP heat-only all-branch four-outcome PC2 comprehensive report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "This report consolidates the participant-level ROI/direct-anatomy reanalysis in which the task-defined BA3a/BA3c-transition predictors were replaced by matched HC19 heat-only values for Branch B primary LNM/BCB, Branch A LNM/BCB sensitivities and the Branch C BCB sensitivity, using CPSP status, NPSI total, saved QST PC2 and saved QST PC1. It contains every TSV result and audit table generated in the analysis directory, full parallel-resampling outputs, quality-control records, provenance, output inventories and links to diagnostic plot files.",
        "",
        "Thermal detection, NPSI evoked pain, temporal summation/WUR and DN4 are excluded from the new principal four-outcome multiplicity families. Randomise and PALM were not rerun.",
        "",
        "## Run overview",
        "",
        f"- Slurm shards per resampling mode: {shards}",
        f"- Bootstrap repetitions per registered model: {bootstrap_reps}",
        f"- Freedman-Lane permutations per registered continuous model: {permutation_reps}",
        f"- Registered bootstrap models: {registered_bootstrap}",
        f"- Registered permutation models: {registered_permutation}",
        f"- Registered leave-one-out models: {registered_loo}",
        f"- Required merge/QC failures: {len(failures)}",
        f"- TSV files embedded: {len(table_paths)}",
        f"- Total embedded TSV data rows, including long-form tables: {total_table_rows}",
        f"- Automatically indexed potentially notable rows: {len(headline_rows)}",
        "",
        "## Interpretation safeguard",
        "",
        "The analysis uses the saved PC2 orientation exactly as stored and does not recompute, rotate or sign-reverse the PCA. Confirm the original numeric loading table before manuscript-level interpretation. The automatic headline index is a navigation aid, not a substitute for model-specific interpretation.",
        "",
    ]

    with md_path.open("w", encoding="utf-8") as md:
        md.write("\n".join(summary_lines))
        md.write("## Final quality control\n\n")
        md.write(markdown_tsv_block(qc_fields, qc_rows))
        md.write("\n## Automatically indexed headline rows\n\n")
        md.write(markdown_tsv_block(headline_fields, headline_rows))
        for category in category_order:
            paths = category_map.get(category, [])
            if not paths:
                continue
            md.write(f"\n# {category}\n\n")
            for path in paths:
                fields, rows = read_tsv(path)
                md.write(f"## {path.relative_to(out_dir)}\n\n")
                md.write(f"Rows: {len(rows)}; columns: {len(fields)}.\n\n")
                md.write(markdown_tsv_block(fields, rows))
        if text_paths or flag_paths:
            md.write("\n# Text audit records and flags\n\n")
            for path in text_paths + flag_paths:
                md.write(f"## {path.relative_to(out_dir)}\n\n```text\n")
                md.write(path.read_text(encoding="utf-8", errors="replace"))
                if not path.read_text(encoding="utf-8", errors="replace").endswith("\n"):
                    md.write("\n")
                md.write("```\n\n")
        md.write("\n# Diagnostic plots and non-tabular outputs\n\n")
        if plot_paths:
            for path in plot_paths:
                md.write(f"- `{path.relative_to(out_dir)}` ({path.stat().st_size} bytes)\n")
        else:
            md.write("No plot files were present.\n")

    style = """
    body{font-family:Arial,Helvetica,sans-serif;max-width:1500px;margin:30px auto;padding:0 18px;line-height:1.45;color:#1f2933}
    h1,h2,h3{color:#102a43} h1{border-bottom:3px solid #bcccdc;padding-bottom:.25rem} h2{margin-top:2rem}
    .summary{background:#f0f4f8;border:1px solid #bcccdc;padding:1rem 1.25rem;border-radius:8px}
    .warning{background:#fffbea;border-left:5px solid #f0b429;padding:.8rem 1rem;margin:1rem 0}
    details{margin:.8rem 0;border:1px solid #d9e2ec;border-radius:6px;background:#fff} summary{cursor:pointer;font-weight:700;padding:.7rem;background:#f5f7fa}
    .meta{padding:0 .8rem;color:#52606d}.tablewrap{overflow:auto;max-height:780px;border-top:1px solid #d9e2ec}
    table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:max-content;min-width:100%} th,td{border:1px solid #d9e2ec;padding:4px 7px;vertical-align:top}
    th{position:sticky;top:0;background:#e4e7eb;z-index:2} tr:nth-child(even){background:#f8fafc}
    code,pre{font-family:Consolas,Menlo,monospace}.toc a{text-decoration:none}.pass{color:#087f5b;font-weight:bold}.fail{color:#c92a2a;font-weight:bold}
    """
    with html_path.open("w", encoding="utf-8") as page:
        page.write("<!doctype html><html><head><meta charset='utf-8'>")
        page.write("<meta name='viewport' content='width=device-width,initial-scale=1'>")
        page.write("<title>CPSP PC2 comprehensive report</title><style>")
        page.write(style)
        page.write("</style></head><body>")
        page.write("<h1>CPSP four-outcome PC2 comprehensive single report</h1>")
        page.write("<div class='summary'>")
        page.write(f"<p><strong>Generated:</strong> {html.escape(datetime.now().astimezone().isoformat(timespec='seconds'))}</p>")
        page.write("<p>This report embeds every TSV result and audit table in the completed analysis directory, including deterministic models, sensitivity analyses, bootstrap, Freedman-Lane permutation, leave-one-out outputs, quality control and provenance.</p>")
        page.write("<ul>")
        for item in summary_lines[10:24]:
            if item.startswith("- "):
                page.write(f"<li>{html.escape(item[2:])}</li>")
        page.write("</ul></div>")
        page.write("<div class='warning'><strong>PC2 safeguard:</strong> saved PC2 scores are used without recomputation or sign reversal. Confirm the original numeric loading table before manuscript interpretation. Automatically indexed rows are navigation aids only.</div>")
        page.write("<h2>Contents</h2><div class='toc'><ul>")
        page.write("<li><a href='#qc'>Final quality control</a></li><li><a href='#headline'>Automatically indexed headline rows</a></li>")
        for idx, category in enumerate(category_order):
            if category_map.get(category):
                page.write(f"<li><a href='#cat{idx}'>{html.escape(category)}</a></li>")
        page.write("<li><a href='#textrecords'>Text audit records and flags</a></li><li><a href='#plots'>Diagnostic plots and non-tabular outputs</a></li></ul></div>")

        page.write("<h2 id='qc'>Final quality control</h2>")
        page.write(html_table(qc_fields, qc_rows))
        page.write("<h2 id='headline'>Automatically indexed headline rows</h2>")
        page.write("<p>Rows are included when an adjusted value is below 0.05, a confidence/significance flag is true, or—if neither applies—a nominal p-value is below 0.05.</p>")
        page.write(html_table(headline_fields, headline_rows))

        for idx, category in enumerate(category_order):
            paths = category_map.get(category, [])
            if not paths:
                continue
            page.write(f"<h1 id='cat{idx}'>{html.escape(category)}</h1>")
            for path in paths:
                fields, rows = read_tsv(path)
                rel = path.relative_to(out_dir)
                page.write(f"<details><summary>{html.escape(str(rel))} — {len(rows)} rows × {len(fields)} columns</summary>")
                page.write(f"<p class='meta'>Authoritative source: <code>{html.escape(str(rel))}</code></p>")
                page.write(html_table(fields, rows))
                page.write("</details>")

        page.write("<h1 id='textrecords'>Text audit records and flags</h1>")
        for path in text_paths + flag_paths:
            rel = path.relative_to(out_dir)
            content = path.read_text(encoding="utf-8", errors="replace")
            page.write(f"<details><summary>{html.escape(str(rel))}</summary><pre>{html.escape(content)}</pre></details>")
        page.write("<h1 id='plots'>Diagnostic plots and non-tabular outputs</h1><ul>")
        if plot_paths:
            for path in plot_paths:
                rel_from_report = os.path.relpath(path, report_dir)
                page.write(f"<li><a href='{html.escape(rel_from_report)}'>{html.escape(str(path.relative_to(out_dir)))}</a> ({path.stat().st_size} bytes)</li>")
        else:
            page.write("<li>No plot files were present.</li>")
        page.write("</ul></body></html>")
    return html_path, md_path

def main() -> int:
    if len(sys.argv) != 5:
        fail("Usage: cpsp_pc2_merge_report.py <analysis_outdir> <shard_count> <bootstrap_reps> <permutation_reps>")
    out_dir = Path(sys.argv[1]).resolve()
    shards = int(sys.argv[2])
    bootstrap_reps = int(sys.argv[3])
    permutation_reps = int(sys.argv[4])
    if not out_dir.is_dir():
        fail(f"Analysis output directory does not exist: {out_dir}")

    audit_dir = out_dir / "audit"
    tables_dir = out_dir / "tables"
    audit_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    registry_path = audit_dir / "PARALLEL_MODEL_REGISTRY.tsv"
    registry_fields, registry_rows = read_tsv(registry_path)
    if not registry_fields or not registry_rows:
        fail(f"Model registry is missing or empty: {registry_path}")
    if "model_id" not in registry_fields:
        fail("Model registry has no model_id column")

    mode_results: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    completion_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for mode in ("bootstrap", "permutation", "loo"):
        flag_name = f"resample_{mode}"
        expected_models = [row for row in registry_rows if as_bool(row.get(flag_name, ""))]
        shard_tables: list[tuple[list[str], list[dict[str, str]]]] = []
        for shard in range(shards):
            table_path = out_dir / "parallel" / mode / f"shard_{shard:03d}.tsv"
            success_path = out_dir / "parallel" / mode / "success" / f"shard_{shard:03d}.ok"
            table_exists = table_path.is_file()
            success_exists = success_path.is_file()
            table_nonempty = table_exists and table_path.stat().st_size > 0
            passed = table_exists and success_exists and table_nonempty
            completion_rows.append({
                "mode": mode,
                "shard_id": shard,
                "table_exists": str(table_exists).upper(),
                "success_flag_exists": str(success_exists).upper(),
                "table_nonempty": str(table_nonempty).upper(),
                "pass": str(passed).upper(),
            })
            if not passed:
                failure_rows.append({
                    "failure_type": "missing_or_incomplete_shard",
                    "mode": mode,
                    "shard_id": shard,
                    "model_id": "",
                    "detail": str(table_path),
                })
            shard_tables.append(read_tsv(table_path))
        combined_fields, combined_rows = combine_tables(shard_tables)
        mode_results[mode] = (combined_fields, combined_rows)
        for expected in expected_models:
            model_id = expected.get("model_id", "")
            model_rows = [row for row in combined_rows if row.get("model_id", "") == model_id]
            observed = len(model_rows)
            if mode == "loo":
                valid = [row for row in model_rows if math.isfinite(as_float(row.get("estimate"))) and normalise_status(row.get("status")) == "OK"]
                valid_fraction = len(valid) / observed if observed else 0.0
                passed = observed >= 15 and valid_fraction >= 0.90
            else:
                valid = [row for row in model_rows if normalise_status(row.get("status")) == "OK"]
                valid_fraction = len(valid) / observed if observed else 0.0
                passed = observed >= 1 and bool(valid)
            coverage_rows.append({
                "mode": mode,
                "registry_index": expected.get("registry_index", ""),
                "model_id": model_id,
                "expected": "TRUE",
                "rows_observed": observed,
                "valid_fraction": f"{valid_fraction:.12g}",
                "pass": str(passed).upper(),
            })
            if not passed:
                failure_rows.append({
                    "failure_type": "model_not_successfully_covered",
                    "mode": mode,
                    "shard_id": "",
                    "model_id": model_id,
                    "detail": f"rows_observed={observed};valid_fraction={valid_fraction:.4g}",
                })

    completion_fields = ["mode", "shard_id", "table_exists", "success_flag_exists", "table_nonempty", "pass"]
    coverage_fields = ["mode", "registry_index", "model_id", "expected", "rows_observed", "valid_fraction", "pass"]
    failure_fields = ["failure_type", "mode", "shard_id", "model_id", "detail"]
    write_tsv(audit_dir / "ARRAY_COMPLETION_AUDIT.tsv", completion_fields, completion_rows)
    write_tsv(audit_dir / "ARRAY_MODEL_COVERAGE_AUDIT.tsv", coverage_fields, coverage_rows)
    write_tsv(audit_dir / "ARRAY_REQUIRED_FAILURES.tsv", failure_fields, failure_rows)

    bootstrap_fields, bootstrap_rows = mode_results["bootstrap"]
    for field in ("valid_fraction", "interval_excludes_zero", "interval_excludes_one_OR"):
        if field not in bootstrap_fields:
            bootstrap_fields.append(field)
    for row in bootstrap_rows:
        requested = as_float(row.get("bootstrap_reps_requested"))
        valid = as_float(row.get("bootstrap_reps_valid"))
        row["valid_fraction"] = f"{valid / max(requested, 1):.12g}" if math.isfinite(valid) and math.isfinite(requested) else ""
        low = as_float(row.get("bootstrap_low")); high = as_float(row.get("bootstrap_high"))
        row["interval_excludes_zero"] = str(math.isfinite(low) and math.isfinite(high) and (low > 0 or high < 0)).upper()
        if as_bool(row.get("binary")):
            low_or = as_float(row.get("bootstrap_OR_low")); high_or = as_float(row.get("bootstrap_OR_high"))
            row["interval_excludes_one_OR"] = str(math.isfinite(low_or) and math.isfinite(high_or) and (low_or > 1 or high_or < 1)).upper()
        else:
            row["interval_excludes_one_OR"] = ""
    write_tsv(tables_dir / "PARALLEL_BOOTSTRAP_ALL_MODELS.tsv", bootstrap_fields, bootstrap_rows)

    permutation_fields, permutation_rows = mode_results["permutation"]
    for field in ("q_BH_within_registered_family", "Holm_within_registered_family", "q_BH_global_transparency", "Holm_global_transparency"):
        if field not in permutation_fields:
            permutation_fields.append(field)
    add_group_adjustment(permutation_rows, "permutation_p_two_sided", "permutation_family", "q_BH_within_registered_family", "BH")
    add_group_adjustment(permutation_rows, "permutation_p_two_sided", "permutation_family", "Holm_within_registered_family", "holm")
    global_values = [row.get("permutation_p_two_sided", "") for row in permutation_rows]
    for row, bh_value, holm_value in zip(permutation_rows, bh_adjust(global_values), holm_adjust(global_values)):
        row["q_BH_global_transparency"] = bh_value
        row["Holm_global_transparency"] = holm_value
    write_tsv(tables_dir / "PARALLEL_FREEDMAN_LANE_PERMUTATION_ALL_MODELS.tsv", permutation_fields, permutation_rows)

    loo_fields, loo_rows = mode_results["loo"]
    write_tsv(tables_dir / "PARALLEL_LOO_LONG_ALL_MODELS.tsv", loo_fields, loo_rows)
    grouped_loo: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in loo_rows:
        grouped_loo[row.get("model_id", "")].append(row)
    loo_summary_fields = [
        "model_id", "dataset", "outcome", "predictor", "binary", "full_estimate", "loo_rows", "valid_fits", "valid_fraction",
        "estimate_median", "estimate_min", "estimate_max", "direction_retained_fraction", "nominal_p_lt_0p05_fraction", "maximum_absolute_estimate_change",
    ]
    loo_summary_rows: list[dict[str, object]] = []
    for model_id, rows in grouped_loo.items():
        estimates = [as_float(row.get("estimate")) for row in rows]
        valid_estimates = [value for value, row in zip(estimates, rows) if math.isfinite(value) and normalise_status(row.get("status")) == "OK"]
        p_values = [as_float(row.get("p_value")) for row in rows]
        valid_p = [value for value in p_values if math.isfinite(value)]
        full_candidates = [as_float(row.get("full_estimate")) for row in rows]
        full_estimate = next((value for value in full_candidates if math.isfinite(value)), math.nan)
        first = rows[0]
        loo_summary_rows.append({
            "model_id": model_id,
            "dataset": first.get("dataset", ""),
            "outcome": first.get("outcome", ""),
            "predictor": first.get("predictor", ""),
            "binary": first.get("binary", ""),
            "full_estimate": fmt_number(full_estimate, 12),
            "loo_rows": len(rows),
            "valid_fits": len(valid_estimates),
            "valid_fraction": f"{len(valid_estimates) / len(rows):.12g}" if rows else "",
            "estimate_median": fmt_number(statistics.median(valid_estimates) if valid_estimates else math.nan, 12),
            "estimate_min": fmt_number(min(valid_estimates) if valid_estimates else math.nan, 12),
            "estimate_max": fmt_number(max(valid_estimates) if valid_estimates else math.nan, 12),
            "direction_retained_fraction": f"{sum(math.copysign(1, value) == math.copysign(1, full_estimate) for value in valid_estimates) / len(valid_estimates):.12g}" if valid_estimates and math.isfinite(full_estimate) and full_estimate != 0 else "",
            "nominal_p_lt_0p05_fraction": f"{sum(value < 0.05 for value in valid_p) / len(valid_p):.12g}" if valid_p else "",
            "maximum_absolute_estimate_change": fmt_number(max(abs(value - full_estimate) for value in valid_estimates) if valid_estimates and math.isfinite(full_estimate) else math.nan, 12),
        })
    write_tsv(tables_dir / "PARALLEL_LOO_SUMMARY_ALL_MODELS.tsv", loo_summary_fields, loo_summary_rows)

    all_pc2_tables: list[tuple[list[str], list[dict[str, str]]]] = []
    all_pc2_source_rows: list[dict[str, str]] = []
    all_pc2_fields: list[str] = ["source_table"]
    seen_pc2_fields = {"source_table"}
    for path in sorted(tables_dir.glob("*.tsv")):
        if path.name == "ALL_PC2_RESULT_ROWS.tsv":
            continue
        fields, rows = read_tsv(path)
        char_key_fields = fields
        for source in rows:
            key = " | ".join(source.get(field, "") for field in char_key_fields)
            if re.search(r"QST_PC2|PC2", key, flags=re.I):
                row = {"source_table": path.name}
                row.update(source)
                all_pc2_source_rows.append(row)
                for field in row:
                    if field not in seen_pc2_fields:
                        seen_pc2_fields.add(field)
                        all_pc2_fields.append(field)
    write_tsv(tables_dir / "ALL_PC2_RESULT_ROWS.tsv", all_pc2_fields, all_pc2_source_rows)

    current_table_paths = sorted(tables_dir.glob("*.tsv")) + sorted(audit_dir.glob("*.tsv"))
    headline_fields, headline_rows = build_headline_index(current_table_paths, out_dir)
    write_tsv(tables_dir / "COMPREHENSIVE_HEADLINE_INDEX.tsv", headline_fields, headline_rows)

    inventory_fields = ["relative_path", "file_type", "size_bytes", "sha256"]
    inventory_rows: list[dict[str, object]] = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(out_dir)
        if rel.as_posix() in {"audit/SHA256SUMS.txt", "tables/OUTPUT_FILE_INVENTORY.tsv"}:
            continue
        inventory_rows.append({
            "relative_path": rel.as_posix(),
            "file_type": path.suffix.lower().lstrip(".") or "no_extension",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_tsv(tables_dir / "OUTPUT_FILE_INVENTORY.tsv", inventory_fields, inventory_rows)

    all_completion_pass = bool(completion_rows) and all(as_bool(row.get("pass")) for row in completion_rows)
    all_coverage_pass = bool(coverage_rows) and all(as_bool(row.get("pass")) for row in coverage_rows)
    qc_fields = ["check", "observed", "expected", "pass"]
    qc_rows = [
        {"check": "all_shards_complete", "observed": int(all_completion_pass), "expected": 1, "pass": str(all_completion_pass).upper()},
        {"check": "all_registered_models_covered", "observed": int(all_coverage_pass), "expected": 1, "pass": str(all_coverage_pass).upper()},
        {"check": "required_failure_data_rows", "observed": len(failure_rows), "expected": 0, "pass": str(len(failure_rows) == 0).upper()},
        {"check": "prep_success_flag", "observed": int((out_dir / "PC2_PREP_SUCCESS.flag").is_file()), "expected": 1, "pass": str((out_dir / "PC2_PREP_SUCCESS.flag").is_file()).upper()},
        {"check": "PC2_result_index_nonempty", "observed": len(all_pc2_source_rows), "expected": ">0", "pass": str(len(all_pc2_source_rows) > 0).upper()},
        {"check": "comprehensive_headline_index_written", "observed": int((tables_dir / "COMPREHENSIVE_HEADLINE_INDEX.tsv").is_file()), "expected": 1, "pass": "TRUE"},
        {"check": "output_inventory_written", "observed": int((tables_dir / "OUTPUT_FILE_INVENTORY.tsv").is_file()), "expected": 1, "pass": "TRUE"},
        {"check": "comprehensive_html_report_planned", "observed": 1, "expected": 1, "pass": "TRUE"},
        {"check": "comprehensive_markdown_report_planned", "observed": 1, "expected": 1, "pass": "TRUE"},
        {"check": "randomise_rerun", "observed": 0, "expected": 0, "pass": "TRUE"},
        {"check": "PALM_rerun", "observed": 0, "expected": 0, "pass": "TRUE"},
    ]
    write_tsv(audit_dir / "FINAL_PC2_PARALLEL_QC.tsv", qc_fields, qc_rows)

    html_path, md_path = write_comprehensive_report(
        out_dir, shards, bootstrap_reps, permutation_reps, registry_rows, failure_rows,
        qc_fields, qc_rows, headline_fields, headline_rows,
    )
    if not html_path.is_file() or html_path.stat().st_size == 0 or not md_path.is_file() or md_path.stat().st_size == 0:
        fail("Comprehensive report files were not created")

    legacy_md = out_dir / "report" / "PC2_PARALLEL_REANALYSIS_REPORT.md"
    legacy_html = out_dir / "report" / "PC2_PARALLEL_REANALYSIS_REPORT.html"
    legacy_md.write_text(
        "# CPSP heat-only all-branch PC2 parallel reanalysis\n\nThe comprehensive report is available at `PC2_COMPREHENSIVE_SINGLE_REPORT.md` and `PC2_COMPREHENSIVE_SINGLE_REPORT.html`.\n",
        encoding="utf-8",
    )
    legacy_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='0; url=PC2_COMPREHENSIVE_SINGLE_REPORT.html'></head><body><a href='PC2_COMPREHENSIVE_SINGLE_REPORT.html'>Open comprehensive report</a></body></html>",
        encoding="utf-8",
    )

    all_qc_pass = all(as_bool(row.get("pass")) for row in qc_rows)
    if all_qc_pass:
        (out_dir / "PC2_PARALLEL_SUCCESS.flag").write_text("SUCCESS\n", encoding="utf-8")
        (out_dir / "PC2_MERGE_REPORT_SUCCESS.flag").write_text("SUCCESS\n", encoding="utf-8")
        print(f"SUCCESS OUT={out_dir}")
        print(f"REPORT={html_path}")
        return 0

    (out_dir / "PC2_MERGE_REPORT_COMPLETED_WITH_QC_FAILURES.flag").write_text(
        f"COMPLETED_WITH_QC_FAILURES\nfailures={len(failure_rows)}\n",
        encoding="utf-8",
    )
    print(f"REPORT_WRITTEN_WITH_QC_FAILURES OUT={out_dir}", file=sys.stderr)
    print(f"REPORT={html_path}", file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
SOURCE_CPSP_PC2_MERGE_REPORT_PY

cat > "$WORK/01_prepare_pc2.sh" <<'SOURCE_01_PREPARE_PC2_SH'
#!/usr/bin/env bash
#SBATCH --job-name=CPSPHEATABPREP
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/CPSPHEATAB_PREP_%j.out
#SBATCH --error=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/CPSPHEATAB_PREP_%j.err

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "ERROR: execute this script; do not source it." >&2; return 2; fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$BUNDLE_ROOT" ]] || BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R_SCRIPT="${R_SCRIPT:-${BUNDLE_ROOT}/cpsp_four_outcome_pc2_prepare.R}"
HEAT_BUILDER="${HEAT_BUILDER:-${BUNDLE_ROOT}/build_heat_only_all_branch_main_dataset.py}"
OUT="${OUT:?OUT must be exported by submit_cpsp_pc2_parallel.sh}"
mkdir -p "${PROJECT}/slurm_logs" "$OUT"/{scripts,audit,tables,plots,report,parallel}

fail(){ echo "ERROR: $*" >&2; exit 1; }
latest_dir_with(){
  local base="$1" pattern="$2" required_rel="$3"
  find "$base" -maxdepth 1 -type d -name "$pattern" -exec test -s "{}/${required_rel}" ';' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}
[[ -d "$RUN" ]] || fail "Phase 5 run root does not exist: $RUN"
[[ -s "$R_SCRIPT" ]] || fail "PC2 preparation R script is missing: $R_SCRIPT"
[[ -s "$HEAT_BUILDER" ]] || fail "Heat-only dataset builder is missing: $HEAT_BUILDER"

if [[ -z "${MAIN_DATA:-}" ]]; then
  REVIEWER_ROOT="$(latest_dir_with "$RUN" 'REVIEWER_REVISION_NO_EXTERNAL_NORMATIVE_*' 'tables/analysis_dataset_with_reviewer_metrics.tsv')"
  if [[ -n "$REVIEWER_ROOT" ]]; then MAIN_DATA="${REVIEWER_ROOT}/tables/analysis_dataset_with_reviewer_metrics.tsv"; else MAIN_DATA="${RUN}/thalamo_operculo_cingulate_followup_20260708_091926/models/thalamo_operculo_cingulate_model_dataset.tsv"; fi
fi
[[ -s "$MAIN_DATA" ]] || fail "Main/reviewer model dataset is missing: $MAIN_DATA"
ORIGINAL_MAIN_DATA="$MAIN_DATA"

if [[ -z "${HEAT_TABLE:-}" ]]; then
  HEAT_ROOT="$(latest_dir_with "$RUN" 'HEAT_ONLY_ALL_BRANCH_PREDICTORS_*' 'tables/heat_only_all_branches_model_ready.tsv')"
  [[ -n "$HEAT_ROOT" ]] || fail "No HEAT_ONLY_ALL_BRANCH_PREDICTORS_* output with a model-ready table was found under $RUN"
  [[ -s "$HEAT_ROOT/HEAT_ONLY_ALL_BRANCHES_SUCCESS.flag" ]] || fail "Newest all-branch heat-only predictor output lacks its success flag: $HEAT_ROOT"
  HEAT_TABLE="${HEAT_ROOT}/tables/heat_only_all_branches_model_ready.tsv"
fi
[[ -s "$HEAT_TABLE" ]] || fail "Heat-only all-branch predictor table is missing: $HEAT_TABLE"
HEAT_ROOT="$(dirname "$(dirname "$HEAT_TABLE")")"
[[ -s "$HEAT_ROOT/HEAT_ONLY_ALL_BRANCHES_SUCCESS.flag" ]] || fail "Heat-only all-branch success flag is missing: $HEAT_ROOT/HEAT_ONLY_ALL_BRANCHES_SUCCESS.flag"

if [[ -z "${HEAT_QC:-}" ]]; then
  HEAT_QC="$HEAT_ROOT/audit/heat_only_all_branches_QC.tsv"
fi
[[ -s "$HEAT_QC" ]] || fail "Heat-only all-branch QC table is missing: $HEAT_QC"

HEAT_MAIN_DATA="$OUT/audit/main_dataset_heat_only_all_branches.tsv"
HEAT_REPLACEMENT_AUDIT="$OUT/audit/HEAT_ONLY_ALL_BRANCHES_REPLACEMENT_AUDIT.tsv"
HEAT_ARCHIVED_COLUMNS="$OUT/audit/HEAT_ONLY_ALL_BRANCHES_ARCHIVED_UNION_COLUMNS.tsv"
python3 "$HEAT_BUILDER" \
  --main "$ORIGINAL_MAIN_DATA" \
  --heat-table "$HEAT_TABLE" \
  --heat-qc "$HEAT_QC" \
  --out "$HEAT_MAIN_DATA" \
  --audit "$HEAT_REPLACEMENT_AUDIT" \
  --archived-columns "$HEAT_ARCHIVED_COLUMNS"
cp -p "$HEAT_QC" "$OUT/audit/SOURCE_heat_only_all_branches_QC.tsv"
[[ -s "$HEAT_ROOT/report/HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md" ]] && cp -p "$HEAT_ROOT/report/HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md" "$OUT/audit/SOURCE_HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md"
MAIN_DATA="$HEAT_MAIN_DATA"

if [[ -z "${S1_DATA:-}" ]]; then
  S1_ROOT="$(latest_dir_with "$RUN" 'S1_BA1_HYPOTHESIS_EXTENSION_*' 'statistics/tables/analysis_dataset_with_S1_QST_timing.tsv')"
  if [[ -n "$S1_ROOT" ]]; then S1_DATA="${S1_ROOT}/statistics/tables/analysis_dataset_with_S1_QST_timing.tsv"; else S1_DATA="NONE"; fi
fi
[[ "$S1_DATA" == "NONE" || -s "$S1_DATA" ]] || fail "S1 dataset is missing: $S1_DATA"

if [[ -z "${MOREL_DATA:-}" ]]; then
  MOREL_ROOT="$(latest_dir_with "$RUN" 'MOREL_POSTERIOR_THALAMIC_BA3A3C_FOLLOWUP_*' 'tables/analysis_dataset_with_thalamic_overlap.tsv')"
  if [[ -n "$MOREL_ROOT" ]]; then MOREL_DATA="${MOREL_ROOT}/tables/analysis_dataset_with_thalamic_overlap.tsv"; else MOREL_DATA="NONE"; fi
fi
[[ "$MOREL_DATA" == "NONE" || -s "$MOREL_DATA" ]] || fail "Morel dataset is missing: $MOREL_DATA"

module purge >/dev/null 2>&1 || true
module load r/4.5.2-gcc14.2.0 >/dev/null 2>&1 || module load r >/dev/null 2>&1 || true
command -v Rscript >/dev/null 2>&1 || fail "Rscript unavailable after loading R module"
command -v python3 >/dev/null 2>&1 || fail "python3 unavailable"
for f in "$R_SCRIPT" "$BUNDLE_ROOT/cpsp_pc2_array_worker.R"; do
  Rscript --vanilla -e "invisible(parse(file='${f}')); cat('R syntax parse passed: ${f}\\n')"
done
python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); compile(p.read_text(encoding="utf-8"), str(p), "exec"); print(f"Python syntax compile passed: {p}")' "$BUNDLE_ROOT/cpsp_pc2_merge_report.py"
python3 -m py_compile "$HEAT_BUILDER"

on_error(){ rc=$?; trap - ERR; printf 'FAILED\nexit_code=%s\ntime=%s\nline=%s\ncommand=%s\n' "$rc" "$(date --iso-8601=seconds 2>/dev/null || date)" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}" > "$OUT/PC2_PREP_FAILED.flag"; exit "$rc"; }
trap on_error ERR

cp -p "$BUNDLE_ROOT"/*.R "$BUNDLE_ROOT"/*.sh "$BUNDLE_ROOT"/*.py "$OUT/scripts/" 2>/dev/null || true
cat > "$OUT/audit/PROVENANCE.txt" <<EOF
analysis=CPSP_four_outcome_PC2_heat_only_all_branches_parallel
created=$(date --iso-8601=seconds 2>/dev/null || date)
PROJECT=$PROJECT
RUN=$RUN
ORIGINAL_MAIN_DATA=$ORIGINAL_MAIN_DATA
HEAT_TABLE=$HEAT_TABLE
HEAT_QC=$HEAT_QC
MAIN_DATA=$MAIN_DATA
S1_DATA=$S1_DATA
MOREL_DATA=$MOREL_DATA
BUNDLE_ROOT=$BUNDLE_ROOT
OUT=$OUT
retained_outcomes=CPSP_status_binary;NPSI_total;QST_PC2_saved;QST_PC1_saved
excluded_from_new_families=thermal_detection;NPSI_evoked;temporal_summation_WUR;DN4
primary_task_ROI=HC19_heatOnly_leftS1PlusBA3cTransition_z2p3
primary_branch=BranchB
heat_only_branch_sensitivities=LNM_BranchA;BCB_BranchA;BCB_BranchC
historical_union_task_ROI_predictors_analysed=0
historical_union_task_ROI_predictors_archived=1
PC2_recomputed=0
PC2_sign_reversed=0
randomise_rerun=0
PALM_rerun=0
EOF

export RUN_BOOTSTRAP=0 RUN_FULL_LOO=0 N_CORES="${SLURM_CPUS_PER_TASK:-1}" BOOTSTRAP_REPS=0
Rscript --vanilla "$R_SCRIPT" "$MAIN_DATA" "$S1_DATA" "$MOREL_DATA" "$OUT"
[[ -s "$OUT/PC2_PREP_SUCCESS.flag" ]] || fail "Preparation ended without PC2_PREP_SUCCESS.flag"
[[ -s "$OUT/audit/PARALLEL_MODEL_REGISTRY.tsv" ]] || fail "Parallel registry missing"
if awk 'NR>1 {n++} END {exit(n>0 ? 0 : 1)}' "$OUT/audit/PREP_REQUIRED_SENSITIVITY_FAILURES.tsv"; then
  fail "Preparation audit contains required failure rows"
fi
cat "${OUT}/audit/FINAL_PC2_PREP_QC.tsv"
echo "PREP_SUCCESS OUT=$OUT"
SOURCE_01_PREPARE_PC2_SH

cat > "$WORK/02_array_worker.sh" <<'SOURCE_02_ARRAY_WORKER_SH'
#!/usr/bin/env bash
#SBATCH --job-name=CPSPHEATARR
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/CPSPHEAT_%A_%a.out
#SBATCH --error=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/CPSPHEAT_%A_%a.err

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "ERROR: execute this script; do not source it." >&2; return 2; fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR:-}}"; [[ -n "$BUNDLE_ROOT" ]] || BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:?OUT is required}"; ARRAY_MODE="${ARRAY_MODE:?ARRAY_MODE is required}"; SHARDS="${SHARDS:?SHARDS is required}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-5000}"; PERMUTATION_REPS="${PERMUTATION_REPS:-10000}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
mkdir -p "${PROJECT}/slurm_logs"
[[ -s "$OUT/PC2_PREP_SUCCESS.flag" ]] || { echo "ERROR: prep success flag missing" >&2; exit 1; }
module purge >/dev/null 2>&1 || true
module load r/4.5.2-gcc14.2.0 >/dev/null 2>&1 || module load r >/dev/null 2>&1 || true
command -v Rscript >/dev/null 2>&1 || { echo "ERROR: Rscript unavailable" >&2; exit 1; }
Rscript --vanilla "$BUNDLE_ROOT/cpsp_pc2_array_worker.R" "$ARRAY_MODE" "$OUT" "$SHARD_ID" "$SHARDS" "$BOOTSTRAP_REPS" "$PERMUTATION_REPS"
SOURCE_02_ARRAY_WORKER_SH

cat > "$WORK/03_merge_report.sh" <<'SOURCE_03_MERGE_REPORT_SH'
#!/usr/bin/env bash
#SBATCH --job-name=CPSPHEATMERGE
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/CPSPHEAT_MERGE_%j.out
#SBATCH --error=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/CPSPHEAT_MERGE_%j.err

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "ERROR: execute this script; do not source it." >&2
    return 2
fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$BUNDLE_ROOT" ]] || BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:?OUT is required}"
SHARDS="${SHARDS:?SHARDS is required}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-5000}"
PERMUTATION_REPS="${PERMUTATION_REPS:-10000}"
MERGER="${BUNDLE_ROOT}/cpsp_pc2_merge_report.py"

mkdir -p "${PROJECT}/slurm_logs" "$OUT/audit" "$OUT/report" "$OUT/tables"
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 unavailable" >&2; exit 1; }
[[ -s "$MERGER" ]] || { echo "ERROR: merger missing: $MERGER" >&2; exit 1; }
python3 -m py_compile "$MERGER"

set +e
python3 "$MERGER" "$OUT" "$SHARDS" "$BOOTSTRAP_REPS" "$PERMUTATION_REPS"
MERGE_RC=$?
set -e

REPORT="$OUT/report/PC2_COMPREHENSIVE_SINGLE_REPORT.html"
[[ -s "$REPORT" ]] || { echo "ERROR: comprehensive report was not written: $REPORT" >&2; exit 1; }

find "$OUT" -type f ! -path "$OUT/audit/SHA256SUMS.txt" -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$OUT/audit/SHA256SUMS.txt"
ARCHIVE="${OUT}.tar.gz"
tar -C "$(dirname "$OUT")" -czf "$ARCHIVE" "$(basename "$OUT")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo "MERGE_RC=$MERGE_RC"
echo "OUT=$OUT"
echo "ARCHIVE=$ARCHIVE"
echo "REPORT=$REPORT"

if [[ $MERGE_RC -ne 0 ]]; then
    echo "ERROR: report was written, but strict final QC did not pass. Inspect audit/FINAL_PC2_PARALLEL_QC.tsv and audit/ARRAY_REQUIRED_FAILURES.tsv." >&2
    exit "$MERGE_RC"
fi
[[ -s "$OUT/PC2_PARALLEL_SUCCESS.flag" ]] || { echo "ERROR: merge success flag missing" >&2; exit 1; }
if awk 'NR>1 {n++} END {exit(n>0 ? 0 : 1)}' "$OUT/audit/ARRAY_REQUIRED_FAILURES.tsv"; then
    echo "ERROR: required array failures are present" >&2
    exit 1
fi

echo "COMPLETE"
SOURCE_03_MERGE_REPORT_SH

cat > "$WORK/prepare_heat_only_all_branches.sh" <<'SOURCE_PREPARE_HEAT_ONLY_ALL_BRANCHES_SH'
#!/usr/bin/env bash
#SBATCH --job-name=heatROI_allBranches
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/scratch/users/arnastam/fmri_preproc/logs/heatROI_allBranches_%j.out
#SBATCH --error=/mnt/scratch/users/arnastam/fmri_preproc/logs/heatROI_allBranches_%j.err

set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
MASTER="${MASTER:-${RUN}/MASTER_REBUILT_PART_B_20260710_215017}"

SOURCE_DATASET="${SOURCE_DATASET:-${MASTER}/tables/model_dataset_REBUILT_structured_HCPex_LNM.tsv}"

HEAT_ROI="${HEAT_ROI:-${PROJECT}/derivatives/phase2_HC_localiser/rois/motion_sensitivity_HC19_exclSub09Sub17/HC19_exclSub09Sub17_heatOnly_leftS1PlusBA3cTransition_z2p3_mask.nii.gz}"
UNION_ROI="${UNION_ROI:-${PROJECT}/derivatives/phase2_HC_localiser/rois/HC19_exclSub09Sub17_desc-motionSensitivity_leftS1BA3cTransition_heatORvibration_z2p3_localiser_mask.nii.gz}"

LNM_A_ROOT="${LNM_A_ROOT:-${PROJECT}/derivatives/phase3_stroke/functional_lnm_v3_fmriprepPV_sensitivity}"
LNM_B_ROOT="${LNM_B_ROOT:-${PROJECT}/derivatives/phase3_stroke/functional_lnm_v3_enantiomorphicPV}"
LNM_A_MANIFEST="${LNM_A_MANIFEST:-${MASTER}/audit/LNM_BranchA_map_manifest.tsv}"
LNM_B_MANIFEST="${LNM_B_MANIFEST:-${MASTER}/audit/LNM_BranchB_map_manifest.tsv}"
LNM_A_COVERAGE80="${LNM_A_COVERAGE80:-${LNM_A_ROOT}/normative_connectome/HC19_exclSub09Sub17_coverage80_primary_mask.nii.gz}"
LNM_B_COVERAGE80="${LNM_B_COVERAGE80:-${LNM_B_ROOT}/normative_connectome/HC19_exclSub09Sub17_coverage80_primary_mask.nii.gz}"

BCB_B_DIR="${BCB_B_DIR:-${PROJECT}/derivatives/phase3_stroke/BCB_outputs/PRIMARY_branchB_enantiomorphic_FSLMNI2mm_HCP7T2mm}"
BCB_A_DIR="${BCB_A_DIR:-${PROJECT}/derivatives/phase3_stroke/BCB_outputs/SENSITIVITY_branchA_fmriprep_FSLMNI2mm_HCP7T2mm}"
BCB_C_DIR="${BCB_C_DIR:-${PROJECT}/derivatives/phase3_stroke/BCB_outputs/SENSITIVITY_branchC_enantiomorphic_FSLMNI1mm_HCP7T1mm}"

BCB_WORK_HIST="${BCB_WORK_HIST:-${PROJECT}/derivatives/phase3_stroke/phase5_model_inputs/BCB_BA3a_BA3c_SELECTED_ROI_work}"
BCB_B_HIST_MASK="${BCB_B_HIST_MASK:-${BCB_WORK_HIST}/BCB_BA3a_BA3c_primary_BranchB_2mm_selectedROI_on_BCB_grid.nii.gz}"
BCB_A_HIST_MASK="${BCB_A_HIST_MASK:-${BCB_WORK_HIST}/BCB_BA3a_BA3c_sensitivity_BranchA_2mm_selectedROI_on_BCB_grid.nii.gz}"
BCB_C_HIST_MASK="${BCB_C_HIST_MASK:-${BCB_WORK_HIST}/BCB_BA3a_BA3c_sensitivity_BranchC_1mm_selectedROI_on_BCB_grid.nii.gz}"

EXPECTED_N="${EXPECTED_N:-63}"
REPRO_TOL="${REPRO_TOL:-0.000005}"

APPTAINER_BIN="${APPTAINER_BIN:-/opt/apps/pkg/tools/apptainer/1.3.6/bin/apptainer}"
FMRIPREP_SIF="${FMRIPREP_SIF:-/opt/apps/pkg/applications/containers/fmriprep/25.1.3/fmriprep_25.1.3.sif}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-${RUN}/HEAT_ONLY_ALL_BRANCH_PREDICTORS_${STAMP}}"
mkdir -p "${PROJECT}/logs" "${OUT}"/{audit,masks/LNM_BranchA,masks/LNM_BranchB,masks/BCB_BranchA,masks/BCB_BranchB,masks/BCB_BranchC,tables,report,tmp}

fail() {
  echo "ERROR: $*" >&2
  printf '%s\n' "$*" > "${OUT}/FAILED.flag"
  exit 2
}

required=(
  "$SOURCE_DATASET" "$HEAT_ROI" "$UNION_ROI"
  "$LNM_A_MANIFEST" "$LNM_B_MANIFEST" "$LNM_A_COVERAGE80" "$LNM_B_COVERAGE80"
  "$BCB_A_DIR" "$BCB_B_DIR" "$BCB_C_DIR"
  "$BCB_A_HIST_MASK" "$BCB_B_HIST_MASK" "$BCB_C_HIST_MASK"
  "$APPTAINER_BIN" "$FMRIPREP_SIF"
)
for p in "${required[@]}"; do [[ -e "$p" ]] || fail "Required input not found: $p"; done

module purge || true
module load fsl/6.0.7 2>/dev/null || module load fsl
export FSLOUTPUTTYPE=NIFTI_GZ

make_bcb_manifest() {
  local branch="$1" dir="$2" output="$3"
  printf 'branch\tparticipant_id\tmap_path\n' > "$output"

  mapfile -t maps < <(
    find "$dir" -maxdepth 1 -type f \( -name 'sub-*_lesion.nii.gz' -o -name 'sub-*_lesion.nii' \) | sort
  )
  if [[ ${#maps[@]} -eq 0 ]]; then
    mapfile -t maps < <(
      find "$dir" -maxdepth 1 -type f \( -name 'sub-*.nii.gz' -o -name 'sub-*.nii' \) | sort
    )
  fi
  [[ ${#maps[@]} -eq "$EXPECTED_N" ]] || fail "$branch: expected ${EXPECTED_N} BCB maps in $dir, found ${#maps[@]}"

  local map base pid
  for map in "${maps[@]}"; do
    base="$(basename "$map")"
    pid="$(printf '%s\n' "$base" | grep -oE 'sub-[0-9]+' | head -1 || true)"
    [[ -n "$pid" ]] || fail "$branch: could not parse participant ID from $map"
    printf '%s\t%s\t%s\n' "$branch" "$pid" "$map" >> "$output"
  done
}

BCB_A_MANIFEST="${OUT}/audit/BCB_BranchA_map_manifest.tsv"
BCB_B_MANIFEST="${OUT}/audit/BCB_BranchB_map_manifest.tsv"
BCB_C_MANIFEST="${OUT}/audit/BCB_BranchC_map_manifest.tsv"
make_bcb_manifest BCB_BranchA "$BCB_A_DIR" "$BCB_A_MANIFEST"
make_bcb_manifest BCB_BranchB "$BCB_B_DIR" "$BCB_B_MANIFEST"
make_bcb_manifest BCB_BranchC "$BCB_C_DIR" "$BCB_C_MANIFEST"

printf 'branch\treference_map\thistorical_union_mask\tresampled_union_mask\tresampled_heat_mask\n' > "${OUT}/audit/BCB_mask_preparation_manifest.tsv"

prepare_bcb_branch() {
  local branch="$1" manifest="$2" hist="$3" outdir="$4"
  local ref union_raw union_mask heat_raw heat_mask
  ref="$(awk -F'\t' 'NR==2 {print $3; exit}' "$manifest")"
  [[ -f "$ref" ]] || fail "$branch: reference map not resolved from $manifest"

  union_raw="${OUT}/tmp/${branch}_union_raw.nii.gz"
  union_mask="${outdir}/heatORvibration_union_on_${branch}_grid.nii.gz"
  heat_raw="${OUT}/tmp/${branch}_heat_raw.nii.gz"
  heat_mask="${outdir}/heatOnly_on_${branch}_grid.nii.gz"

  flirt -in "$UNION_ROI" -ref "$ref" -applyxfm -usesqform -interp nearestneighbour -out "$union_raw"
  fslmaths "$union_raw" -thr 0.5 -bin "$union_mask"
  flirt -in "$HEAT_ROI" -ref "$ref" -applyxfm -usesqform -interp nearestneighbour -out "$heat_raw"
  fslmaths "$heat_raw" -thr 0.5 -bin "$heat_mask"
  rm -f "$union_raw" "$heat_raw"

  read -r hvox _ <<< "$(fslstats "$heat_mask" -V)"
  [[ "${hvox:-0}" -gt 0 ]] || fail "$branch: heat-only mask became empty on BCB grid"

  printf '%s\t%s\t%s\t%s\t%s\n' "$branch" "$ref" "$hist" "$union_mask" "$heat_mask" >> "${OUT}/audit/BCB_mask_preparation_manifest.tsv"
}

prepare_bcb_branch BCB_BranchA "$BCB_A_MANIFEST" "$BCB_A_HIST_MASK" "${OUT}/masks/BCB_BranchA"
prepare_bcb_branch BCB_BranchB "$BCB_B_MANIFEST" "$BCB_B_HIST_MASK" "${OUT}/masks/BCB_BranchB"
prepare_bcb_branch BCB_BranchC "$BCB_C_MANIFEST" "$BCB_C_HIST_MASK" "${OUT}/masks/BCB_BranchC"

cat > "${OUT}/audit/input_paths.tsv" <<EOF
input\tpath
source_dataset\t${SOURCE_DATASET}
heat_source_roi\t${HEAT_ROI}
union_source_roi\t${UNION_ROI}
lnm_branchA_manifest\t${LNM_A_MANIFEST}
lnm_branchB_manifest\t${LNM_B_MANIFEST}
lnm_branchA_coverage80\t${LNM_A_COVERAGE80}
lnm_branchB_coverage80\t${LNM_B_COVERAGE80}
bcb_branchA_directory\t${BCB_A_DIR}
bcb_branchB_directory\t${BCB_B_DIR}
bcb_branchC_directory\t${BCB_C_DIR}
bcb_branchA_historical_union_mask\t${BCB_A_HIST_MASK}
bcb_branchB_historical_union_mask\t${BCB_B_HIST_MASK}
bcb_branchC_historical_union_mask\t${BCB_C_HIST_MASK}
EOF

export PROJECT RUN MASTER SOURCE_DATASET HEAT_ROI UNION_ROI OUT EXPECTED_N REPRO_TOL
export LNM_A_MANIFEST LNM_B_MANIFEST LNM_A_COVERAGE80 LNM_B_COVERAGE80
export BCB_A_MANIFEST BCB_B_MANIFEST BCB_C_MANIFEST
export BCB_A_HIST_MASK BCB_B_HIST_MASK BCB_C_HIST_MASK

"${APPTAINER_BIN}" exec \
  -B /mnt/scratch:/mnt/scratch \
  -B /opt/apps:/opt/apps \
  "${FMRIPREP_SIF}" python - <<'PY'
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.stats import pearsonr, spearmanr

OUT = Path(os.environ["OUT"])
SOURCE_DATASET = Path(os.environ["SOURCE_DATASET"])
HEAT_ROI = Path(os.environ["HEAT_ROI"])
UNION_ROI = Path(os.environ["UNION_ROI"])
EXPECTED_N = int(os.environ["EXPECTED_N"])
REPRO_TOL = float(os.environ["REPRO_TOL"])

AUDIT = OUT / "audit"
MASKS = OUT / "masks"
TABLES = OUT / "tables"
REPORT = OUT / "report"
for d in (AUDIT, MASKS, TABLES, REPORT):
    d.mkdir(parents=True, exist_ok=True)

def fail(message: str) -> None:
    (OUT / "FAILED.flag").write_text(message + "\n")
    raise RuntimeError(message)

def delimiter(path: Path) -> str:
    text = path.read_text(errors="replace")[:8192]
    return "\t" if text.count("\t") >= text.count(",") else ","

def read_rows(path: Path):
    with path.open("r", newline="", errors="replace") as f:
        return list(csv.DictReader(f, delimiter=delimiter(path)))

def write_rows(path: Path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    if not fields:
        fields = ["no_rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

def pid(value) -> str:
    m = re.search(r"(?:sub[-_ ]*)?(\d{1,3})", str(value or ""), flags=re.I)
    return f"sub-{int(m.group(1)):02d}" if m else str(value or "").strip()

def first_col(rows, names):
    if not rows:
        return None
    keys = list(rows[0])
    lower = {x.lower(): x for x in keys}
    for name in names:
        if name in keys:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None

def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan

def same_grid(a, b, atol=1e-3):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol)

def binary(img):
    x = np.asanyarray(img.dataobj)
    return np.isfinite(x) & (x > 0)

def overlap(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = int((a & b).sum())
    na, nb = int(a.sum()), int(b.sum())
    union = int((a | b).sum())
    return {
        "a_voxels": na,
        "b_voxels": nb,
        "intersection_voxels": inter,
        "dice": 2 * inter / (na + nb) if na + nb else np.nan,
        "jaccard": inter / union if union else np.nan,
        "exact_binary_match": bool(np.array_equal(a, b)),
    }

def summarise(values):
    x = np.asarray(values, float)
    f = x[np.isfinite(x)]
    if not len(f):
        return dict(n_finite=0, n_unique=0, mean=np.nan, sd=np.nan, min=np.nan, max=np.nan)
    return dict(
        n_finite=int(len(f)),
        n_unique=int(len(np.unique(np.round(f, 12)))),
        mean=float(f.mean()),
        sd=float(f.std(ddof=1)) if len(f) > 1 else np.nan,
        min=float(f.min()),
        max=float(f.max()),
    )

def corrs(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    out = {}
    if keep.sum() >= 3 and np.std(x[keep]) > 1e-12 and np.std(y[keep]) > 1e-12:
        out["pearson_r"] = float(pearsonr(x[keep], y[keep]).statistic)
        out["spearman_rho"] = float(spearmanr(x[keep], y[keep]).statistic)
    else:
        out["pearson_r"] = np.nan
        out["spearman_rho"] = np.nan
    return out

def map_manifest(path: Path, label: str):
    rows = read_rows(path)
    pcol = first_col(rows, ["participant_id", "subject_id", "subject", "participant"])
    fcol = first_col(rows, ["map_path", "path", "file", "fisherz_map_path", "disconnectome_map_path"])
    if pcol is None or fcol is None:
        fail(f"{label}: participant/path columns not identified in {path}")
    result = []
    seen = set()
    for row in rows:
        p = pid(row.get(pcol))
        f = Path(str(row.get(fcol, "")).strip())
        if p in seen:
            fail(f"{label}: duplicate participant {p}")
        if not f.is_file():
            fail(f"{label}: map missing for {p}: {f}")
        seen.add(p)
        result.append((p, f))
    result.sort()
    if len(result) != EXPECTED_N:
        fail(f"{label}: expected {EXPECTED_N} maps, found {len(result)}")
    return result

rows = read_rows(SOURCE_DATASET)
pcol = first_col(rows, ["participant_id", "subject_id", "subject"])
if pcol is None:
    fail("Participant ID column not found in source dataset")
meta = {}
for row in rows:
    p = pid(row.get(pcol))
    if p:
        row = dict(row); row["participant_id"] = p
        if p in meta:
            fail(f"Duplicate participant in source dataset: {p}")
        meta[p] = row
if len(meta) != EXPECTED_N:
    fail(f"Expected {EXPECTED_N} source participants, found {len(meta)}")
participants = sorted(meta)

heat_img = nib.load(str(HEAT_ROI))
union_img = nib.load(str(UNION_ROI))

lnm_configs = [
    {
        "label": "LNM_BranchA",
        "manifest": Path(os.environ["LNM_A_MANIFEST"]),
        "coverage": Path(os.environ["LNM_A_COVERAGE80"]),
        "frozen": "LNM_BA3a_BA3c_sensitivity_BranchA",
        "heat_raw": "LNM_BA3a_BA3c_sensitivity_BranchA_heatOnly",
        "heat_z": "z_LNM_BA3a_BA3c_sensitivity_BranchA_heatOnly",
    },
    {
        "label": "LNM_BranchB",
        "manifest": Path(os.environ["LNM_B_MANIFEST"]),
        "coverage": Path(os.environ["LNM_B_COVERAGE80"]),
        "frozen": "LNM_BA3a_BA3c_primary_BranchB",
        "heat_raw": "LNM_BA3a_BA3c_primary_BranchB_heatOnly",
        "heat_z": "z_LNM_BA3a_BA3c_primary_BranchB_heatOnly",
    },
]

bcb_mask_manifest = {r["branch"]: r for r in read_rows(AUDIT / "BCB_mask_preparation_manifest.tsv")}
bcb_configs = [
    {
        "label": "BCB_BranchA",
        "manifest": Path(os.environ["BCB_A_MANIFEST"]),
        "hist": Path(os.environ["BCB_A_HIST_MASK"]),
        "frozen": "BCB_BA3a_BA3c_sensitivity_BranchA_2mm",
        "heat_raw": "BCB_BA3a_BA3c_sensitivity_BranchA_2mm_heatOnly",
        "heat_z": "z_BCB_BA3a_BA3c_sensitivity_BranchA_2mm_heatOnly",
    },
    {
        "label": "BCB_BranchB",
        "manifest": Path(os.environ["BCB_B_MANIFEST"]),
        "hist": Path(os.environ["BCB_B_HIST_MASK"]),
        "frozen": "BCB_BA3a_BA3c_primary_BranchB_2mm",
        "heat_raw": "BCB_BA3a_BA3c_primary_BranchB_2mm_heatOnly",
        "heat_z": "z_BCB_BA3a_BA3c_primary_BranchB_2mm_heatOnly",
    },
    {
        "label": "BCB_BranchC",
        "manifest": Path(os.environ["BCB_C_MANIFEST"]),
        "hist": Path(os.environ["BCB_C_HIST_MASK"]),
        "frozen": "BCB_BA3a_BA3c_sensitivity_BranchC_1mm",
        "heat_raw": "BCB_BA3a_BA3c_sensitivity_BranchC_1mm_heatOnly",
        "heat_z": "z_BCB_BA3a_BA3c_sensitivity_BranchC_1mm_heatOnly",
    },
]

all_values = {p: {"participant_id": p} for p in participants}
qc_rows = []
comparison_rows = []
mask_rows = []

for cfg in lnm_configs:
    maps = map_manifest(cfg["manifest"], cfg["label"])
    if [p for p, _ in maps] != participants:
        fail(f"{cfg['label']}: participants do not match source dataset")
    reference = nib.load(str(maps[0][1]))
    coverage_img = nib.load(str(cfg["coverage"]))
    if not same_grid(reference, coverage_img):
        fail(f"{cfg['label']}: coverage80 mask does not match map grid")
    coverage = coverage_img.get_fdata(dtype=np.float32) > 0.5

    def prepared_weights(source_img, name):
        rs = source_img if same_grid(source_img, reference) else resample_from_to(source_img, reference, order=1)
        w = rs.get_fdata(dtype=np.float32)
        w[~np.isfinite(w)] = 0
        w = np.clip(w, 0, None)
        w *= coverage.astype(np.float32)
        support = w > 0.0001
        if support.sum() < 1 or w.sum() <= 0:
            fail(f"{cfg['label']}: {name} ROI empty after historical preparation")
        path = MASKS / cfg["label"] / f"{name}_weighted.nii.gz"
        h = reference.header.copy(); h.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(w.astype(np.float32), reference.affine, h), str(path))
        mask_rows.append({
            "modality": "LNM", "branch": cfg["label"], "mask": name,
            "support_voxels": int(support.sum()), "weight_sum": float(w.sum()), "path": str(path)
        })
        return w, support

    union_w, union_s = prepared_weights(union_img, "heatORvibration_union")
    heat_w, heat_s = prepared_weights(heat_img, "heatOnly")

    union_values = {}; heat_values = {}
    for p, f in maps:
        img = nib.load(str(f))
        if not same_grid(img, reference):
            fail(f"{cfg['label']}: geometry mismatch for {p}")
        data = np.asanyarray(img.dataobj).astype(np.float32)
        for name, weights, support, target in (
            ("union", union_w, union_s, union_values),
            ("heat", heat_w, heat_s, heat_values),
        ):
            vals = data[support]; ww = weights[support]
            keep = np.isfinite(vals) & np.isfinite(ww) & (ww > 0)
            if not keep.any():
                fail(f"{cfg['label']}: no finite {name} values for {p}")
            target[p] = float(np.average(vals[keep], weights=ww[keep]))

    diffs = []
    for p in participants:
        frozen = finite(meta[p].get(cfg["frozen"]))
        if not math.isfinite(frozen):
            fail(f"{cfg['label']}: frozen calibration value missing for {p}")
        diffs.append(abs(union_values[p] - frozen))
        all_values[p][cfg["heat_raw"]] = heat_values[p]
        comparison_rows.append({
            "participant_id": p, "modality": "LNM", "branch": cfg["label"],
            "frozen_union": frozen, "reextracted_union": union_values[p],
            "heat_only": heat_values[p], "union_absolute_difference": abs(union_values[p] - frozen)
        })

    heat_summary = summarise([heat_values[p] for p in participants])
    c = corrs([meta[p][cfg["frozen"]] for p in participants], [heat_values[p] for p in participants])
    max_diff = max(diffs)
    reproduction = max_diff <= REPRO_TOL
    usable = heat_summary["n_finite"] == EXPECTED_N and heat_summary["n_unique"] >= 10 and heat_summary["sd"] > 1e-8
    verdict = "GREEN" if reproduction and usable else "RED"
    qc_rows.append({
        "modality": "LNM", "branch": cfg["label"], "frozen_column": cfg["frozen"],
        "heat_column": cfg["heat_raw"], "union_reproduction_max_abs_diff": max_diff,
        "union_reproduction_tolerance": REPRO_TOL, "union_reproduction_pass": reproduction,
        **heat_summary, **c, "participants_with_nonzero": "NA", "mask_exact_match": "NA", "verdict": verdict
    })

for cfg in bcb_configs:
    maps = map_manifest(cfg["manifest"], cfg["label"])
    if [p for p, _ in maps] != participants:
        fail(f"{cfg['label']}: participants do not match source dataset")
    reference = nib.load(str(maps[0][1]))
    hist_img = nib.load(str(cfg["hist"]))
    row = bcb_mask_manifest.get(cfg["label"])
    if row is None:
        fail(f"{cfg['label']}: mask-preparation manifest row missing")
    union_img_bcb = nib.load(row["resampled_union_mask"])
    heat_img_bcb = nib.load(row["resampled_heat_mask"])
    for img, name in ((hist_img, "historical"), (union_img_bcb, "resampled_union"), (heat_img_bcb, "heat")):
        if not same_grid(img, reference):
            fail(f"{cfg['label']}: {name} mask does not match BCB map grid")

    hist_mask = binary(hist_img)
    union_mask = binary(union_img_bcb)
    heat_mask = binary(heat_img_bcb)
    mask_match = overlap(union_mask, hist_mask)
    mask_rows.extend([
        {"modality": "BCB", "branch": cfg["label"], "mask": "historical_union", "support_voxels": int(hist_mask.sum()), "weight_sum": int(hist_mask.sum()), "path": str(cfg["hist"])},
        {"modality": "BCB", "branch": cfg["label"], "mask": "heatOnly", "support_voxels": int(heat_mask.sum()), "weight_sum": int(heat_mask.sum()), "path": row["resampled_heat_mask"]},
    ])
    if heat_mask.sum() < 1:
        fail(f"{cfg['label']}: heat-only mask is empty")

    def extract(mask):
        values = {}
        details = {}
        roi_n = int(mask.sum())
        for p, f in maps:
            img = nib.load(str(f))
            if not same_grid(img, reference):
                fail(f"{cfg['label']}: map geometry mismatch for {p}")
            vals = np.asanyarray(img.dataobj).astype(float)[mask]
            if not np.isfinite(vals).all() or len(vals) != roi_n:
                fail(f"{cfg['label']}: non-finite or incomplete ROI values for {p}")
            nz = vals[vals != 0]
            values[p] = float(nz.mean()) if len(nz) else 0.0
            details[p] = int(len(nz))
        return values, details

    union_values, _ = extract(hist_mask)
    heat_values, heat_nz = extract(heat_mask)
    diffs = []
    for p in participants:
        frozen = finite(meta[p].get(cfg["frozen"]))
        if not math.isfinite(frozen):
            fail(f"{cfg['label']}: frozen calibration value missing for {p}")
        diffs.append(abs(union_values[p] - frozen))
        all_values[p][cfg["heat_raw"]] = heat_values[p]
        comparison_rows.append({
            "participant_id": p, "modality": "BCB", "branch": cfg["label"],
            "frozen_union": frozen, "reextracted_union": union_values[p],
            "heat_only": heat_values[p], "union_absolute_difference": abs(union_values[p] - frozen),
            "heat_nonzero_voxels": heat_nz[p]
        })

    heat_summary = summarise([heat_values[p] for p in participants])
    c = corrs([meta[p][cfg["frozen"]] for p in participants], [heat_values[p] for p in participants])
    max_diff = max(diffs)
    reproduction = max_diff <= REPRO_TOL
    n_any = sum(heat_nz[p] > 0 for p in participants)
    mask_exact = bool(mask_match["exact_binary_match"])
    usable = heat_summary["n_finite"] == EXPECTED_N and heat_summary["n_unique"] >= 10 and heat_summary["sd"] > 1e-8 and n_any >= math.ceil(0.75 * EXPECTED_N)
    verdict = "GREEN" if reproduction and mask_exact and usable else "RED"
    qc_rows.append({
        "modality": "BCB", "branch": cfg["label"], "frozen_column": cfg["frozen"],
        "heat_column": cfg["heat_raw"], "union_reproduction_max_abs_diff": max_diff,
        "union_reproduction_tolerance": REPRO_TOL, "union_reproduction_pass": reproduction,
        **heat_summary, **c, "participants_with_nonzero": n_any,
        "mask_exact_match": mask_exact, "mask_dice": mask_match["dice"], "verdict": verdict
    })

for cfg in lnm_configs + bcb_configs:
    raw = np.array([all_values[p][cfg["heat_raw"]] for p in participants], float)
    mean = raw.mean(); sd = raw.std(ddof=1)
    if not math.isfinite(sd) or sd <= 0:
        fail(f"Cannot z-score {cfg['heat_raw']}")
    for p, value in zip(participants, raw):
        all_values[p][cfg["heat_z"]] = float((value - mean) / sd)

ordered = ["participant_id"]
for cfg in [lnm_configs[1], lnm_configs[0], bcb_configs[1], bcb_configs[0], bcb_configs[2]]:
    ordered.extend([cfg["heat_raw"], cfg["heat_z"]])
model_rows = [all_values[p] for p in participants]
write_rows(TABLES / "heat_only_all_branches_model_ready.tsv", model_rows, ordered)
write_rows(AUDIT / "heat_only_all_branches_QC.tsv", qc_rows)
write_rows(AUDIT / "heat_only_vs_union_participant_comparison.tsv", comparison_rows)
write_rows(AUDIT / "heat_only_all_branches_mask_summary.tsv", mask_rows)

all_green = all(row["verdict"] == "GREEN" for row in qc_rows)
report = [
    "# Heat-only all-branch predictor preparation",
    "",
    "This stage prepares predictors only. It does not test clinical outcomes.",
    "",
    "## Prepared measures",
    "",
    "- LNM Branch B: enantiomorphic weighted-PV primary",
    "- LNM Branch A: fMRIPrep weighted-PV sensitivity",
    "- BCB Branch B: enantiomorphic 2 mm primary",
    "- BCB Branch A: fMRIPrep 2 mm sensitivity",
    "- BCB Branch C: enantiomorphic 1 mm sensitivity",
    "",
    "## Quality-control results",
    "",
    "| Modality | Branch | Union reproduction | Mask match | N finite | N unique | Non-zero participants | Pearson heat vs union | Verdict |",
    "|---|---|---:|---:|---:|---:|---:|---:|---|",
]
for q in qc_rows:
    report.append(
        f"| {q['modality']} | {q['branch']} | {q['union_reproduction_pass']} "
        f"(max diff {q['union_reproduction_max_abs_diff']:.3g}) | {q.get('mask_exact_match', 'NA')} | "
        f"{q['n_finite']} | {q['n_unique']} | {q.get('participants_with_nonzero', 'NA')} | "
        f"{q['pearson_r']:.5f} | **{q['verdict']}** |"
    )
report += [
    "",
    "## Decision",
    "",
    ("**GREEN: all five heat-only predictors are technically suitable for the replacement analysis.**" if all_green
     else "**RED: at least one branch failed. Do not start the replacement analysis until the failed branch is resolved.**"),
    "",
    "## Model-ready table",
    "",
    "`tables/heat_only_all_branches_model_ready.tsv`",
    "",
    "The table contains separate raw and z-scored heat-only columns. Historical heat-or-vibration values are not overwritten at this stage.",
]
(REPORT / "HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md").write_text("\n".join(report) + "\n")

if not all_green:
    fail("One or more all-branch heat-only predictor QC gates failed")

(OUT / "HEAT_ONLY_ALL_BRANCHES_SUCCESS.flag").write_text("PASS\n")
print("PASS: all five heat-only branch predictors prepared")
print(f"OUT={OUT}")
print(f"TABLE={TABLES / 'heat_only_all_branches_model_ready.tsv'}")
print(f"REPORT={REPORT / 'HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md'}")
PY

sha256sum \
  "${OUT}/tables/heat_only_all_branches_model_ready.tsv" \
  "${OUT}/audit/heat_only_all_branches_QC.tsv" \
  "${OUT}/report/HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md" \
  > "${OUT}/audit/key_output_SHA256SUMS.txt"

cat <<EOF
============================================================
Heat-only all-branch predictor preparation completed.
OUT=${OUT}
Model-ready values:
  ${OUT}/tables/heat_only_all_branches_model_ready.tsv
QC:
  ${OUT}/audit/heat_only_all_branches_QC.tsv
Report:
  ${OUT}/report/HEAT_ONLY_ALL_BRANCH_PREDICTOR_REPORT.md
============================================================
EOF
SOURCE_PREPARE_HEAT_ONLY_ALL_BRANCHES_SH

cat > "$WORK/submit_cpsp_pc2_parallel.sh" <<'SOURCE_SUBMIT_CPSP_PC2_PARALLEL_SH'
#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "ERROR: execute this script; do not source it." >&2; return 2; fi
set -Eeuo pipefail
umask 0027
PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARDS="${SHARDS:-64}"; MAX_CONCURRENT_PER_ARRAY="${MAX_CONCURRENT_PER_ARRAY:-32}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-5000}"; PERMUTATION_REPS="${PERMUTATION_REPS:-10000}"
PREP_CPUS="${PREP_CPUS:-16}"; PREP_MEM="${PREP_MEM:-64G}"; PREP_TIME="${PREP_TIME:-1-00:00:00}"
ARRAY_MEM="${ARRAY_MEM:-12G}"; ARRAY_TIME="${ARRAY_TIME:-2-00:00:00}"
MERGE_MEM="${MERGE_MEM:-32G}"; MERGE_TIME="${MERGE_TIME:-08:00:00}"
PARTITION="${PARTITION:-nodes}"
[[ -d "$RUN" ]] || { echo "ERROR: RUN does not exist: $RUN" >&2; exit 1; }
mkdir -p "$PROJECT/slurm_logs"
chmod u+x "$BUNDLE_ROOT"/*.sh "$BUNDLE_ROOT"/*.R "$BUNDLE_ROOT"/*.py 2>/dev/null || true
python3 "$BUNDLE_ROOT/validate_bundle.py"
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch unavailable; run this on Barkla." >&2; exit 1; }
module purge >/dev/null 2>&1 || true
module load r/4.5.2-gcc14.2.0 >/dev/null 2>&1 || module load r >/dev/null 2>&1 || true
command -v Rscript >/dev/null 2>&1 || { echo "ERROR: Rscript unavailable after loading the R module." >&2; exit 1; }
for f in "$BUNDLE_ROOT"/*.R; do Rscript --vanilla -e "invisible(parse(file='${f}')); cat('R parse passed: ${f}\n')"; done
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-${RUN}/CPSP_FOUR_OUTCOME_PC2_HEAT_ONLY_ALL_BRANCHES_PARALLEL_${STAMP}}"
mkdir -p "$OUT"/{audit,tables,plots,report,parallel,scripts}
COMMON="ALL,PROJECT=$PROJECT,RUN=$RUN,BUNDLE_ROOT=$BUNDLE_ROOT,OUT=$OUT,SHARDS=$SHARDS,BOOTSTRAP_REPS=$BOOTSTRAP_REPS,PERMUTATION_REPS=$PERMUTATION_REPS"
[[ -n "${MAIN_DATA:-}" ]] && COMMON+=",MAIN_DATA=$MAIN_DATA"
[[ -n "${S1_DATA:-}" ]] && COMMON+=",S1_DATA=$S1_DATA"
[[ -n "${MOREL_DATA:-}" ]] && COMMON+=",MOREL_DATA=$MOREL_DATA"
[[ -n "${HEAT_TABLE:-}" ]] && COMMON+=",HEAT_TABLE=$HEAT_TABLE"
[[ -n "${HEAT_QC:-}" ]] && COMMON+=",HEAT_QC=$HEAT_QC"
PREP_JOB="$(sbatch --parsable --partition="$PARTITION" --cpus-per-task="$PREP_CPUS" --mem="$PREP_MEM" --time="$PREP_TIME" --chdir="$BUNDLE_ROOT" --export="$COMMON" "$BUNDLE_ROOT/01_prepare_pc2.sh")"
BOOT_JOB="$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=1 --mem="$ARRAY_MEM" --time="$ARRAY_TIME" --array="0-$((SHARDS-1))%${MAX_CONCURRENT_PER_ARRAY}" --dependency="afterok:${PREP_JOB}" --chdir="$BUNDLE_ROOT" --export="$COMMON,ARRAY_MODE=bootstrap" "$BUNDLE_ROOT/02_array_worker.sh")"
PERM_JOB="$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=1 --mem="$ARRAY_MEM" --time="$ARRAY_TIME" --array="0-$((SHARDS-1))%${MAX_CONCURRENT_PER_ARRAY}" --dependency="afterok:${PREP_JOB}" --chdir="$BUNDLE_ROOT" --export="$COMMON,ARRAY_MODE=permutation" "$BUNDLE_ROOT/02_array_worker.sh")"
LOO_JOB="$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=1 --mem="$ARRAY_MEM" --time="$ARRAY_TIME" --array="0-$((SHARDS-1))%${MAX_CONCURRENT_PER_ARRAY}" --dependency="afterok:${PREP_JOB}" --chdir="$BUNDLE_ROOT" --export="$COMMON,ARRAY_MODE=loo" "$BUNDLE_ROOT/02_array_worker.sh")"
MERGE_JOB="$(sbatch --parsable --partition="$PARTITION" --cpus-per-task=2 --mem="$MERGE_MEM" --time="$MERGE_TIME" --dependency="afterok:${BOOT_JOB}:${PERM_JOB}:${LOO_JOB}" --chdir="$BUNDLE_ROOT" --export="$COMMON" "$BUNDLE_ROOT/03_merge_report.sh")"
cat > "$OUT/audit/SUBMISSION_MANIFEST.txt" <<EOF
submitted=$(date --iso-8601=seconds 2>/dev/null || date)
OUT=$OUT
PREP_JOB=$PREP_JOB
BOOTSTRAP_ARRAY_JOB=$BOOT_JOB
PERMUTATION_ARRAY_JOB=$PERM_JOB
LOO_ARRAY_JOB=$LOO_JOB
MERGE_JOB=$MERGE_JOB
SHARDS=$SHARDS
MAX_CONCURRENT_PER_ARRAY=$MAX_CONCURRENT_PER_ARRAY
BOOTSTRAP_REPS=$BOOTSTRAP_REPS
PERMUTATION_REPS=$PERMUTATION_REPS
EOF
cat <<EOF
Submitted CPSP heat-only all-branch PC2 parallel workflow.
OUT: $OUT
Prep: $PREP_JOB
Bootstrap array: $BOOT_JOB
Permutation array: $PERM_JOB
LOO array: $LOO_JOB
Merge/report: $MERGE_JOB

Live overview:
  squeue -j ${PREP_JOB},${BOOT_JOB},${PERM_JOB},${LOO_JOB},${MERGE_JOB} -o '%.18i %.9P %.18j %.2t %.10M %.10l %.6D %R'
Detailed accounting:
  sacct -j ${PREP_JOB},${BOOT_JOB},${PERM_JOB},${LOO_JOB},${MERGE_JOB} --format=JobID,JobName,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
Monitor helper:
  bash $BUNDLE_ROOT/monitor_cpsp_pc2_parallel.sh $OUT
EOF
SOURCE_SUBMIT_CPSP_PC2_PARALLEL_SH

cat > "$WORK/validate_bundle.py" <<'SOURCE_VALIDATE_BUNDLE_PY'
#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "cpsp_four_outcome_pc2_prepare.R",
    "cpsp_pc2_array_worker.R",
    "cpsp_pc2_merge_report.py",
    "build_heat_only_all_branch_main_dataset.py",
    "prepare_heat_only_all_branches.sh",
    "01_prepare_pc2.sh",
    "02_array_worker.sh",
    "03_merge_report.sh",
    "submit_cpsp_pc2_parallel.sh",
    "install_and_submit_cpsp_pc2_parallel.sh",
    "monitor_cpsp_pc2_parallel.sh",
    "rerun_merge_only.sh",
    "README.md",
    "INSTALL_AND_RUN.txt",
    "CHANGELOG.md",
]
missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))

def strip_strings_and_comments(text: str, language: str) -> str:
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                out.extend("  ")
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(" ")
            i += 1
            continue
        if ch == "#":
            while i < len(text) and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        out.append(ch)
        i += 1
    if quote is not None:
        raise SystemExit(f"Unclosed string literal in {language} source")
    return "".join(out)

def check_r(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    clean = strip_strings_and_comments(text, path.name)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[tuple[str, int]] = []
    line = 1
    for ch in clean:
        if ch == "\n":
            line += 1
        elif ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack or stack[-1][0] != pairs[ch]:
                raise SystemExit(f"Delimiter mismatch in {path.name} at line {line}: {ch}")
            stack.pop()
    if stack:
        raise SystemExit(f"Unclosed delimiter in {path.name}: {stack[-1]}")

for name in [n for n in REQUIRED if n.endswith(".R")]:
    check_r(ROOT / name)

for name in [n for n in REQUIRED if n.endswith(".py")]:
    path = ROOT / name
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as exc:
        raise SystemExit(f"Python compile failed for {name}: {exc}") from exc

for name in [n for n in REQUIRED if n.endswith(".sh")]:
    proc = subprocess.run(["bash", "-n", str(ROOT / name)], text=True, capture_output=True)
    if proc.returncode:
        raise SystemExit(f"bash -n failed for {name}:\n{proc.stderr}")

prep = (ROOT / "cpsp_four_outcome_pc2_prepare.R").read_text(encoding="utf-8")
worker = (ROOT / "cpsp_pc2_array_worker.R").read_text(encoding="utf-8")
merge = (ROOT / "cpsp_pc2_merge_report.py").read_text(encoding="utf-8")
submit = (ROOT / "submit_cpsp_pc2_parallel.sh").read_text(encoding="utf-8")
prepare_shell = (ROOT / "01_prepare_pc2.sh").read_text(encoding="utf-8")
merge_shell = (ROOT / "03_merge_report.sh").read_text(encoding="utf-8")
heat_builder = (ROOT / "build_heat_only_all_branch_main_dataset.py").read_text(encoding="utf-8")

markers = {
    "saved PC2 alias resolution": 'c("QST_PC2_model", "QST_PC2", "qst_pc2", "PC2")',
    "four retained outcomes": 'outcome = c("CPSP_status_binary", "NPSI_total_four", "QST_PC2_four", "QST_PC1_four")',
    "parallel registry": 'PARALLEL_MODEL_REGISTRY.tsv',
    "PC2 provenance": 'QST_PC2_PROVENANCE.tsv',
    "prep completion audit": 'PREP_SENSITIVITY_COMPLETION_AUDIT.tsv',
    "prep failure table": 'PREP_REQUIRED_SENSITIVITY_FAILURES.tsv',
    "prep success flag": 'PC2_PREP_SUCCESS.flag',
}
for label, marker in markers.items():
    if marker not in prep:
        raise SystemExit(f"Missing implementation marker ({label}): {marker}")

if re.search(r'sink\s*\([^\n]*type\s*=\s*["\']message["\'][^\n]*split\s*=\s*TRUE', prep, flags=re.I):
    raise SystemExit("Unsupported split=TRUE message sink remains in the PC2 preparation script")
if 'sink(log_path, type = "output", split = TRUE)' not in prep:
    raise SystemExit("Expected safe output-only split sink is missing")
if 'type = "message"' in prep or "type = 'message'" in prep:
    raise SystemExit("Preparation script should leave warnings/messages in Slurm stderr rather than sink them")
if "QST_thermal_detection_four" in prep:
    raise SystemExit("Thermal detection remains as an active four-outcome variable in PC2 preparation script")
if re.search(r'outcome\s*=\s*c\([^\n]*thermal', prep, flags=re.I):
    raise SystemExit("Thermal detection appears in an active outcome vector")
if 'invisible(parse(file=' not in prepare_shell or 'invisible(parse(file=' not in submit:
    raise SystemExit("Quiet Barkla-side R parse gates are missing")
if "RUN_BOOTSTRAP=0 RUN_FULL_LOO=0" not in prepare_shell:
    raise SystemExit("Preparation job does not disable inherited bootstrap and LOO")
if "NR>1" not in prepare_shell or "NR>1" not in merge_shell:
    raise SystemExit("Header-only failure-table guard is missing")
if "--array=\"0-$((SHARDS-1))%${MAX_CONCURRENT_PER_ARRAY}\"" not in submit:
    raise SystemExit("Expected bounded Slurm array declaration is missing")
for mode in ("bootstrap", "permutation", "loo"):
    if f"ARRAY_MODE={mode}" not in submit:
        raise SystemExit(f"Submission for {mode} array is missing")
if "afterok:${BOOT_JOB}:${PERM_JOB}:${LOO_JOB}" not in submit:
    raise SystemExit("Merge dependency does not require all three arrays")
for marker in ("Freedman", "permutation_p_two_sided", "bootstrap_reps_valid", "omitted_id"):
    if marker not in worker:
        raise SystemExit(f"Array worker marker missing: {marker}")

for marker in (
    "LNM_BA3a_BA3c_primary_BranchB_heatOnly",
    "LNM_BA3a_BA3c_sensitivity_BranchA_heatOnly",
    "BCB_BA3a_BA3c_primary_BranchB_2mm_heatOnly",
    "BCB_BA3a_BA3c_sensitivity_BranchA_2mm_heatOnly",
    "BCB_BA3a_BA3c_sensitivity_BranchC_1mm_heatOnly",
    "heat_only_all_branches_replacement",
    "historical_union_predictors_archived",
):
    if marker not in heat_builder:
        raise SystemExit(f"Heat-only all-branch builder marker missing: {marker}")
if "HEAT_ONLY_ALL_BRANCHES_REPLACEMENT_AUDIT.tsv" not in prepare_shell:
    raise SystemExit("Heat-only all-branch replacement audit is not wired into preparation")
if "HEAT_ONLY_ALL_BRANCHES_SUCCESS.flag" not in prepare_shell:
    raise SystemExit("All-branch predictor success flag is not required by preparation")
if "CPSP_FOUR_OUTCOME_PC2_HEAT_ONLY_ALL_BRANCHES_PARALLEL_" not in submit:
    raise SystemExit("Heat-only all-branch output prefix is missing")
for marker in (
    "BA3a_BA3c_heat_only_task_defined",
    "primary_heat_only_branch_sensitivity_four",
    "BranchA_heat_only",
    "BranchC_heat_only",
    "heat_only_all_branches_replacement",
):
    if marker not in prep:
        raise SystemExit(f"Heat-only all-branch R marker missing: {marker}")
if "historical_union_NOT_HEAT_ONLY" in prep:
    raise SystemExit("Historical-union primary branch sensitivity labels remain in the replacement analysis")

for marker in ("ARRAY_REQUIRED_FAILURES.tsv", "PARALLEL_FREEDMAN_LANE_PERMUTATION_ALL_MODELS.tsv", "ALL_PC2_RESULT_ROWS.tsv", "PC2_COMPREHENSIVE_SINGLE_REPORT.html", "COMPREHENSIVE_HEADLINE_INDEX.tsv"):
    if marker not in merge:
        raise SystemExit(f"Merge/report marker missing: {marker}")

entries: list[str] = []
for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append(f"{digest}  ./{path.relative_to(ROOT).as_posix()}")
(ROOT / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

print("Bundle validation passed")
print(f"Root: {ROOT}")
print(f"Files checksummed: {len(entries)}")
print("R lexical delimiter/string checks: passed")
print("Python compile checks: passed")
print("Bash syntax checks: passed")
print("Note: definitive R parse is enforced on Barkla before the prep analysis runs")
SOURCE_VALIDATE_BUNDLE_PY

cat > "$WORK/build_morel_posterior_sensory_targets_v1.sh" <<'SOURCE_BUILD_MOREL_POSTERIOR_SENSORY_TARGETS_V1_SH'
#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "ERROR: do not source this script."
    echo "Run: bash ${BASH_SOURCE[0]}"
    return 2
fi

set -Eeuo pipefail
export LC_ALL=C
export LANG=C

PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
ATLAS_ROOT="${ATLAS_ROOT:-${PROJECT}/resources/atlases/Morel_MNI152_v1.0/unpacked}"
LEFT="${ATLAS_ROOT}/left-vols-1mm"
RIGHT="${ATLAS_ROOT}/right-vols-1mm"
OUT="${OUT:-${PROJECT}/resources/atlases/Morel_MNI152_v1.0/derived_posterior_sensory_targets_v1}"

mkdir -p "$OUT"/{components,masks,audit}

module purge >/dev/null 2>&1 || true
module load fsl/6.0.7 >/dev/null 2>&1 || module load fsl >/dev/null 2>&1 || true

for cmd in fslmaths fslstats fslval sha256sum; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "ERROR: required command unavailable: $cmd" >&2
        exit 1
    }
done

required=(
    "$LEFT/VPLa.nii.gz"
    "$LEFT/VPLp.nii.gz"
    "$LEFT/VPI.nii.gz"
    "$LEFT/PuA.nii.gz"
    "$RIGHT/VPLa.nii.gz"
    "$RIGHT/VPLp.nii.gz"
    "$RIGHT/VPI.nii.gz"
    "$RIGHT/PuA.nii.gz"
)

for path in "${required[@]}"; do
    [[ -f "$path" ]] || {
        echo "ERROR: Morel component missing: $path" >&2
        exit 1
    }
done

reference="$LEFT/VPLa.nii.gz"
ref_dim="$(fslval "$reference" dim1)x$(fslval "$reference" dim2)x$(fslval "$reference" dim3)"
ref_pix="$(fslval "$reference" pixdim1)x$(fslval "$reference" pixdim2)x$(fslval "$reference" pixdim3)"
ref_sform="$(fslval "$reference" sform_code)"
ref_qform="$(fslval "$reference" qform_code)"

{
    echo -e "component\tpath\tdimensions\tvoxel_size_mm\tsform_code\tqform_code\tminimum\tmaximum\tnonzero_voxels\tvolume_mm3\tgeometry_matches_reference"
    for path in "${required[@]}"; do
        name="$(basename "$(dirname "$path")")/$(basename "$path")"
        dim="$(fslval "$path" dim1)x$(fslval "$path" dim2)x$(fslval "$path" dim3)"
        pix="$(fslval "$path" pixdim1)x$(fslval "$path" pixdim2)x$(fslval "$path" pixdim3)"
        sform="$(fslval "$path" sform_code)"
        qform="$(fslval "$path" qform_code)"
        range="$(fslstats "$path" -R)"
        minimum="$(awk '{print $1}' <<< "$range")"
        maximum="$(awk '{print $2}' <<< "$range")"
        vv="$(fslstats "$path" -V)"
        voxels="$(awk '{print $1}' <<< "$vv")"
        volume="$(awk '{print $2}' <<< "$vv")"
        match=FALSE
        if [[ "$dim" == "$ref_dim" && "$pix" == "$ref_pix" && "$sform" == "$ref_sform" && "$qform" == "$ref_qform" ]]; then
            match=TRUE
        fi
        echo -e "${name}\t${path}\t${dim}\t${pix}\t${sform}\t${qform}\t${minimum}\t${maximum}\t${voxels}\t${volume}\t${match}"
    done
} > "$OUT/audit/source_component_geometry.tsv"

if grep -q $'\tFALSE$' "$OUT/audit/source_component_geometry.tsv"; then
    echo "ERROR: one or more Morel component masks differ from the reference geometry." >&2
    cat "$OUT/audit/source_component_geometry.tsv" >&2
    exit 1
fi

for side in left right; do
    src="$LEFT"
    [[ "$side" == "right" ]] && src="$RIGHT"

    for nucleus in VPLa VPLp VPI PuA; do
        fslmaths "$src/${nucleus}.nii.gz" -bin \
            "$OUT/components/${side}_${nucleus}_bin.nii.gz"
    done
done

fslmaths "$OUT/components/left_VPLa_bin.nii.gz" \
    -add "$OUT/components/left_VPLp_bin.nii.gz" \
    -add "$OUT/components/right_VPLa_bin.nii.gz" \
    -add "$OUT/components/right_VPLp_bin.nii.gz" \
    -bin "$OUT/masks/Morel_VPL_VPLa_plus_VPLp_bilateral_1mm.nii.gz"

fslmaths "$OUT/components/left_PuA_bin.nii.gz" \
    -add "$OUT/components/right_PuA_bin.nii.gz" \
    -bin "$OUT/masks/Morel_PuA_bilateral_1mm.nii.gz"

fslmaths "$OUT/components/left_VPI_bin.nii.gz" \
    -add "$OUT/components/right_VPI_bin.nii.gz" \
    -bin "$OUT/masks/Morel_VPI_bilateral_1mm.nii.gz"

fslmaths "$OUT/components/left_VPLp_bin.nii.gz" \
    -add "$OUT/components/left_VPI_bin.nii.gz" \
    -add "$OUT/components/left_PuA_bin.nii.gz" \
    -add "$OUT/components/right_VPLp_bin.nii.gz" \
    -add "$OUT/components/right_VPI_bin.nii.gz" \
    -add "$OUT/components/right_PuA_bin.nii.gz" \
    -bin "$OUT/masks/Morel_posterior_sensory_borderzone_VPLp_VPI_PuA_bilateral_1mm.nii.gz"

fslmaths "$OUT/components/left_VPLp_bin.nii.gz" \
    -add "$OUT/components/right_VPLp_bin.nii.gz" \
    -bin "$OUT/masks/Morel_VPLp_bilateral_1mm.nii.gz"

cat > "$OUT/audit/target_definitions.tsv" <<EOF
target	definition	role	VMpo_equivalence
VPL	Morel VPLa union VPLp	principal lateral sensory thalamic target	none
PuA	Morel anterior pulvinar	principal posterior thalamic target	none
posterior_sensory_borderzone	Morel VPLp union VPI union PuA	principal anatomically motivated border-zone composite	not equivalent to VMpo
VPI	Morel ventral posterior inferior	supplementary adjacent-nucleus comparator	none
VPLp	Morel posterior VPL	descriptive localisation component	none
EOF

{
    echo -e "target\tpath\tnonzero_voxels\tvolume_mm3\tminimum\tmaximum\tbinary"
    for path in "$OUT"/masks/*.nii.gz; do
        target="$(basename "$path" .nii.gz)"
        vv="$(fslstats "$path" -V)"
        voxels="$(awk '{print $1}' <<< "$vv")"
        volume="$(awk '{print $2}' <<< "$vv")"
        range="$(fslstats "$path" -R)"
        minimum="$(awk '{print $1}' <<< "$range")"
        maximum="$(awk '{print $2}' <<< "$range")"
        binary=FALSE
        if [[ "$minimum" == "0.000000" && "$maximum" == "1.000000" && "$voxels" -gt 0 ]]; then
            binary=TRUE
        fi
        echo -e "${target}\t${path}\t${voxels}\t${volume}\t${minimum}\t${maximum}\t${binary}"
    done
} > "$OUT/audit/derived_mask_geometry_and_volume.tsv"

if grep -q $'\tFALSE$' "$OUT/audit/derived_mask_geometry_and_volume.tsv"; then
    echo "ERROR: one or more derived masks are empty or non-binary." >&2
    cat "$OUT/audit/derived_mask_geometry_and_volume.tsv" >&2
    exit 1
fi

{
    echo -e "sha256\tpath"
    find "$OUT/masks" "$OUT/components" -type f -name '*.nii.gz' -print0 |
    sort -z |
    while IFS= read -r -d '' path; do
        echo -e "$(sha256sum "$path" | awk '{print $1}')\t$path"
    done
} > "$OUT/audit/mask_sha256.tsv"

cat > "$OUT/audit/provenance.txt" <<EOF
Atlas source:
$ATLAS_ROOT

Resolution:
1 mm Morel MNI152 volume masks

Definitions:
VPL = bilateral VPLa + VPLp
PuA = bilateral PuA
VPI = bilateral VPI
posterior_sensory_borderzone = bilateral VPLp + VPI + PuA

The Morel release contains no explicit VMpo label.
No derived target is labelled or treated as VMpo.
EOF

cat > "$OUT/SUCCESS.flag" <<EOF
SUCCESS
completed=$(date --iso-8601=seconds 2>/dev/null || date)
VPL_MASK=$OUT/masks/Morel_VPL_VPLa_plus_VPLp_bilateral_1mm.nii.gz
PUA_MASK=$OUT/masks/Morel_PuA_bilateral_1mm.nii.gz
VPI_MASK=$OUT/masks/Morel_VPI_bilateral_1mm.nii.gz
POSTERIOR_BORDERZONE_MASK=$OUT/masks/Morel_posterior_sensory_borderzone_VPLp_VPI_PuA_bilateral_1mm.nii.gz
EOF

echo "============================================================"
echo "MOREL POSTERIOR-SENSORY TARGETS CREATED"
echo "============================================================"
cat "$OUT/SUCCESS.flag"
echo
column -t -s $'\t' "$OUT/audit/derived_mask_geometry_and_volume.tsv"
SOURCE_BUILD_MOREL_POSTERIOR_SENSORY_TARGETS_V1_SH

cat > "$WORK/run_thalamic_operculo_insular_S2_subregions_LNM_BCB_four_outcomes_v1_1_FIXED.sh" <<'SOURCE_RUN_THALAMIC_OPERCULO_INSULAR_S2_SUBREGIONS_LNM_BCB_FOUR_OUTCOMES_V1_1_FIXED_SH'
#!/usr/bin/env bash
#SBATCH --job-name=THALOPsub4
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/THALOPsub4_%j.out
#SBATCH --error=/mnt/scratch/users/arnastam/fmri_preproc/slurm_logs/THALOPsub4_%j.err

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "ERROR: execute this script; do not source it." >&2
  return 2
fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
FINAL4="${FINAL4:-${RUN}/CPSP_FOUR_OUTCOME_PC2_HEAT_ONLY_ALL_BRANCHES_PARALLEL_20260803_101539}"
if [[ -z "${DATASET:-}" ]]; then
  if [[ -s "${FINAL4}/audit/prepared_main.tsv" ]]; then
    DATASET="${FINAL4}/audit/prepared_main.tsv"
  elif [[ -s "${FINAL4}/audit/main_dataset_heat_only_all_branches.tsv" ]]; then
    DATASET="${FINAL4}/audit/main_dataset_heat_only_all_branches.tsv"
  else
    echo "ERROR: could not resolve final four-outcome prepared dataset under ${FINAL4}/audit" >&2
    exit 1
  fi
fi
MASTER="${MASTER:-${RUN}/MASTER_REBUILT_PART_B_20260710_215017}"
MOREL="${MOREL:-${RUN}/MOREL_POSTERIOR_THALAMIC_BA3A3C_FOLLOWUP_20260711_114247}"
FOLLOWUP="${FOLLOWUP:-${RUN}/thalamo_operculo_cingulate_followup_20260708_091926}"
EXPECTED_N="${EXPECTED_N:-63}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-${RUN}/THALAMIC_OPERCULO_INSULAR_S2_SUBREGION_LNM_BCB_FOUR_OUTCOME_${STAMP}}"
mkdir -p "${PROJECT}/slurm_logs" "${OUT}"/{audit,tables,masks/LNM_BranchA,masks/LNM_BranchB,masks/BCB_BranchA,masks/BCB_BranchB,masks/BCB_BranchC,masks/source_atlas,plots,report,tmp,scripts}

fail(){ echo "ERROR: $*" >&2; exit 1; }
need_file(){ [[ -s "$1" ]] || fail "Required file missing/empty: $1"; }
need_dir(){ [[ -d "$1" ]] || fail "Required directory missing: $1"; }

LNM_A_ROOT="${LNM_A_ROOT:-${PROJECT}/derivatives/phase3_stroke/functional_lnm_v3_fmriprepPV_sensitivity}"
LNM_B_ROOT="${LNM_B_ROOT:-${PROJECT}/derivatives/phase3_stroke/functional_lnm_v3_enantiomorphicPV}"
LNM_A_MANIFEST="${LNM_A_MANIFEST:-${MASTER}/audit/LNM_BranchA_map_manifest.tsv}"
LNM_B_MANIFEST="${LNM_B_MANIFEST:-${MASTER}/audit/LNM_BranchB_map_manifest.tsv}"
LNM_A_COVERAGE80="${LNM_A_COVERAGE80:-${LNM_A_ROOT}/normative_connectome/HC19_exclSub09Sub17_coverage80_primary_mask.nii.gz}"
LNM_B_COVERAGE80="${LNM_B_COVERAGE80:-${LNM_B_ROOT}/normative_connectome/HC19_exclSub09Sub17_coverage80_primary_mask.nii.gz}"

BCB_B_DIR="${BCB_B_DIR:-${PROJECT}/derivatives/phase3_stroke/BCB_outputs/PRIMARY_branchB_enantiomorphic_FSLMNI2mm_HCP7T2mm}"
BCB_A_DIR="${BCB_A_DIR:-${PROJECT}/derivatives/phase3_stroke/BCB_outputs/SENSITIVITY_branchA_fmriprep_FSLMNI2mm_HCP7T2mm}"
BCB_C_DIR="${BCB_C_DIR:-${PROJECT}/derivatives/phase3_stroke/BCB_outputs/SENSITIVITY_branchC_enantiomorphic_FSLMNI1mm_HCP7T1mm}"
HCP7T_2MM="${HCP7T_2MM:-/mnt/scratch/users/arnastam/tools/BCB_atlases/DISCONNECTOME_Package_X_HCP7T_2mm_tracks}"
HCP7T_1MM="${HCP7T_1MM:-/mnt/scratch/users/arnastam/tools/BCB_atlases/DISCONNECTOME_Package_X_HCP7T_1mm_tracks}"

MOREL_MASK_DIR="${MOREL_MASK_DIR:-${MOREL}/masks_on_lesion_grid}"
VPL_MASK="${VPL_MASK:-${MOREL_MASK_DIR}/VPL_on_lesion_grid_bilateral.nii.gz}"
VPI_MASK="${VPI_MASK:-${MOREL_MASK_DIR}/VPI_on_lesion_grid_bilateral.nii.gz}"
PUA_MASK="${PUA_MASK:-${MOREL_MASK_DIR}/PuA_on_lesion_grid_bilateral.nii.gz}"
BORDER_MASK="${BORDER_MASK:-${MOREL_MASK_DIR}/posterior_sensory_borderzone_on_lesion_grid_bilateral.nii.gz}"

APPTAINER_BIN="${APPTAINER_BIN:-/opt/apps/pkg/tools/apptainer/1.3.6/bin/apptainer}"
FMRIPREP_SIF="${FMRIPREP_SIF:-/opt/apps/pkg/applications/containers/fmriprep/25.1.3/fmriprep_25.1.3.sif}"

for f in "$DATASET" "$LNM_A_MANIFEST" "$LNM_B_MANIFEST" "$LNM_A_COVERAGE80" "$LNM_B_COVERAGE80" \
         "$VPL_MASK" "$VPI_MASK" "$PUA_MASK" "$BORDER_MASK" "$FMRIPREP_SIF"; do need_file "$f"; done
for d in "$BCB_A_DIR" "$BCB_B_DIR" "$BCB_C_DIR"; do need_dir "$d"; done
[[ -x "$APPTAINER_BIN" ]] || APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
[[ -n "$APPTAINER_BIN" && -x "$APPTAINER_BIN" ]] || fail "Apptainer/Singularity unavailable"

module purge >/dev/null 2>&1 || true
module load fsl/6.0.7 >/dev/null 2>&1 || module load fsl >/dev/null 2>&1 || true
command -v flirt >/dev/null 2>&1 || fail "FSL flirt unavailable"
command -v fslmaths >/dev/null 2>&1 || fail "FSL fslmaths unavailable"
module load r/4.5.2-gcc14.2.0 >/dev/null 2>&1 || module load r >/dev/null 2>&1 || true
command -v Rscript >/dev/null 2>&1 || fail "Rscript unavailable"
command -v python3 >/dev/null 2>&1 || fail "python3 unavailable"
command -v fslroi >/dev/null 2>&1 || fail "FSL fslroi unavailable"
command -v fslstats >/dev/null 2>&1 || fail "FSL fslstats unavailable"
command -v fslval >/dev/null 2>&1 || fail "FSL fslval unavailable"

if [[ -z "${FSLDIR:-}" ]]; then
  FSLDIR="$(cd "$(dirname "$(command -v fslmaths)")/.." && pwd)"
fi
[[ -d "${FSLDIR}/data/atlases" ]] || fail "Could not resolve FSL atlas directory under FSLDIR=${FSLDIR}"
FSL_ATLAS_ROOT="${FSL_ATLAS_ROOT:-${FSLDIR}/data/atlases}"
HO_XML="${HO_XML:-${FSL_ATLAS_ROOT}/HarvardOxford-Cortical.xml}"
JUELICH_XML="${JUELICH_XML:-${FSL_ATLAS_ROOT}/Juelich.xml}"
HO_PROB="${HO_PROB:-${FSL_ATLAS_ROOT}/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz}"
JUELICH_PROB="${JUELICH_PROB:-${FSL_ATLAS_ROOT}/Juelich/Juelich-prob-1mm.nii.gz}"
[[ -s "$HO_PROB" ]] || HO_PROB="${FSL_ATLAS_ROOT}/HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz"
[[ -s "$JUELICH_PROB" ]] || JUELICH_PROB="${FSL_ATLAS_ROOT}/Juelich/Juelich-prob-2mm.nii.gz"
for f in "$HO_XML" "$JUELICH_XML" "$HO_PROB" "$JUELICH_PROB"; do need_file "$f"; done

ATLAS_MASK_DIR="${OUT}/masks/source_atlas"
OPERC_MASK="${OPERC_MASK:-${ATLAS_MASK_DIR}/opercular_cortex_HO25_bilateral_source.nii.gz}"
INSULA_MASK="${INSULA_MASK:-${ATLAS_MASK_DIR}/insular_cortex_HO25_bilateral_source.nii.gz}"
S2_MASK="${S2_MASK:-${ATLAS_MASK_DIR}/S2_Juelich_OP1_OP4_25pct_bilateral_source.nii.gz}"
ATLAS_LABEL_AUDIT="${OUT}/audit/operculo_insular_S2_atlas_label_audit.tsv"
printf 'target\tatlas_xml\tatlas_probability\tthreshold_percent\tlabel\tvolume_index\tlabel_voxels\n' > "$ATLAS_LABEL_AUDIT"

atlas_index_exact(){
  local xml="$1" label="$2"
  python3 - "$xml" "$label" <<'PY_ATLAS_INDEX'
import sys, xml.etree.ElementTree as ET
xml, wanted = sys.argv[1], sys.argv[2]
norm = lambda x: " ".join((x or "").split())
root = ET.parse(xml).getroot()
hits=[]
for lab in root.iter('label'):
    if norm(lab.text) == norm(wanted):
        hits.append(lab.attrib.get('index'))
if len(hits) != 1 or hits[0] is None:
    raise SystemExit(f"Expected exactly one atlas label match for {wanted!r} in {xml}; found {hits}")
print(int(hits[0]))
PY_ATLAS_INDEX
}

make_union_mask(){
  local target="$1" atlas="$2" xml="$3" outmask="$4"; shift 4
  local dim4 idx label tmp nvox first=1
  dim4="$(fslval "$atlas" dim4)"
  rm -f "$outmask"
  for label in "$@"; do
    idx="$(atlas_index_exact "$xml" "$label")"
    [[ "$idx" =~ ^[0-9]+$ ]] || fail "$target: non-integer atlas index for $label"
    (( idx >= 0 && idx < dim4 )) || fail "$target: atlas index $idx outside 0..$((dim4-1)) for $label"
    tmp="${OUT}/tmp/${target}_atlasvol_${idx}.nii.gz"
    fslroi "$atlas" "$tmp" "$idx" 1
    fslmaths "$tmp" -thr 25 -bin "$tmp"
    nvox="$(fslstats "$tmp" -V | awk '{print $1}')"
    [[ "$nvox" -gt 0 ]] || fail "$target: label became empty at 25%: $label"
    if (( first )); then
      fslmaths "$tmp" -mul 1 "$outmask"
      first=0
    else
      fslmaths "$outmask" -add "$tmp" -bin "$outmask"
    fi
    printf '%s\t%s\t%s\t25\t%s\t%s\t%s\n' "$target" "$xml" "$atlas" "$label" "$idx" "$nvox" >> "$ATLAS_LABEL_AUDIT"
    rm -f "$tmp"
  done
  nvox="$(fslstats "$outmask" -V | awk '{print $1}')"
  [[ "$nvox" -gt 0 ]] || fail "$target: final union mask is empty"
}

make_union_mask "opercular_cortex" "$HO_PROB" "$HO_XML" "$OPERC_MASK" \
  "Central Opercular Cortex" \
  "Parietal Operculum Cortex"
make_union_mask "insular_cortex" "$HO_PROB" "$HO_XML" "$INSULA_MASK" \
  "Insular Cortex"
make_union_mask "S2_OP1_OP4" "$JUELICH_PROB" "$JUELICH_XML" "$S2_MASK" \
  "GM Secondary somatosensory cortex / Parietal operculum OP1 L" \
  "GM Secondary somatosensory cortex / Parietal operculum OP1 R" \
  "GM Secondary somatosensory cortex / Parietal operculum OP2 L" \
  "GM Secondary somatosensory cortex / Parietal operculum OP2 R" \
  "GM Secondary somatosensory cortex / Parietal operculum OP3 L" \
  "GM Secondary somatosensory cortex / Parietal operculum OP3 R" \
  "GM Secondary somatosensory cortex / Parietal operculum OP4 L" \
  "GM Secondary somatosensory cortex / Parietal operculum OP4 R"

cp -p "${BASH_SOURCE[0]}" "${OUT}/scripts/$(basename "${BASH_SOURCE[0]}")"

cat > "${OUT}/audit/PROVENANCE.txt" <<EOF
analysis=thalamic_plus_operculo_insular_S2_subregion_LNM_BCB_four_outcome_localisation
created=$(date --iso-8601=seconds)
PROJECT=${PROJECT}
RUN=${RUN}
FINAL4=${FINAL4}
DATASET=${DATASET}
MASTER=${MASTER}
MOREL=${MOREL}
FOLLOWUP=${FOLLOWUP}
retained_outcomes=CPSP_status_binary;NPSI_total_four;QST_PC1_four;QST_PC2_four
main_internal_branch=BranchB
main_public_label=Main branch
sensitivity_internal_BranchA_public_label=Sensitivity Branch B
sensitivity_internal_BranchC_public_label=Sensitivity Branch C
ROIs=VPL;VPI;PuA;posterior_sensory_borderzone_VPLp_VPI_PuA;opercular_cortex;insular_cortex;S2_OP1_OP4
network_target_laterality=fixed_bilateral
posterior_borderzone_definition=VPLp+VPI+PuA;not_an_atlas_defined_VMpo
opercular_cortex_definition=HarvardOxford_Central_Opercular_plus_Parietal_Operculum;25pct;bilateral
insular_cortex_definition=HarvardOxford_Insular_Cortex;25pct;bilateral
S2_definition=Juelich_OP1_OP2_OP3_OP4_left_plus_right;25pct;bilateral
HO_probability_atlas=${HO_PROB}
Juelich_probability_atlas=${JUELICH_PROB}
OPERC_mask=${OPERC_MASK}
INSULA_mask=${INSULA_MASK}
S2_mask=${S2_MASK}
LNM_BranchA_root=${LNM_A_ROOT}
LNM_BranchB_root=${LNM_B_ROOT}
LNM_BranchA_manifest=${LNM_A_MANIFEST}
LNM_BranchB_manifest=${LNM_B_MANIFEST}
LNM_BranchA_coverage80=${LNM_A_COVERAGE80}
LNM_BranchB_coverage80=${LNM_B_COVERAGE80}
BCB_BranchA_maps=${BCB_A_DIR}
BCB_BranchB_maps=${BCB_B_DIR}
BCB_BranchC_maps=${BCB_C_DIR}
HCP7T_2mm_provenance=${HCP7T_2MM}
HCP7T_1mm_provenance=${HCP7T_1MM}
VPL_mask=${VPL_MASK}
VPI_mask=${VPI_MASK}
PuA_mask=${PUA_MASK}
posterior_borderzone_mask=${BORDER_MASK}
main_covariates=z_log1p_lesion_volume+original_lesion_side_model
FDR_main=q4_within_target_modality;q_within_family_modality;q28_within_modality;q_within_family_all_modalities;q56_all
permutations=NOT_RUN
PALM_randomise=NOT_RUN
EOF

printf 'roi\tdisplay_label\tsource_mask\n' > "${OUT}/audit/target_source_manifest.tsv"
printf 'VPL\tMorel VPL\t%s\n' "$VPL_MASK" >> "${OUT}/audit/target_source_manifest.tsv"
printf 'VPI\tMorel VPI\t%s\n' "$VPI_MASK" >> "${OUT}/audit/target_source_manifest.tsv"
printf 'PuA\tMorel PuA (anterior pulvinar)\t%s\n' "$PUA_MASK" >> "${OUT}/audit/target_source_manifest.tsv"
printf 'posterior_sensory_borderzone\tVMpo-related posterior sensory border-zone (VPLp+VPI+PuA; not atlas-defined VMpo)\t%s\n' "$BORDER_MASK" >> "${OUT}/audit/target_source_manifest.tsv"
printf 'opercular_cortex\tOpercular cortex (Harvard-Oxford Central + Parietal Operculum, 25%% bilateral)\t%s\n' "$OPERC_MASK" >> "${OUT}/audit/target_source_manifest.tsv"
printf 'insular_cortex\tInsular cortex (Harvard-Oxford Insular Cortex, 25%% bilateral)\t%s\n' "$INSULA_MASK" >> "${OUT}/audit/target_source_manifest.tsv"
printf 'S2_OP1_OP4\tS2 / parietal operculum (Juelich OP1-OP4, 25%% bilateral)\t%s\n' "$S2_MASK" >> "${OUT}/audit/target_source_manifest.tsv"

Rscript --vanilla - "$DATASET" "${OUT}/audit/expected_participants.txt" "$EXPECTED_N" <<'RS_IDS'
args <- commandArgs(trailingOnly=TRUE)
d <- read.delim(args[1], check.names=FALSE, stringsAsFactors=FALSE)
stopifnot("participant_id" %in% names(d))
canon <- function(x){ z <- suppressWarnings(as.integer(gsub("[^0-9]", "", as.character(x)))); ifelse(is.na(z), as.character(x), sprintf("sub-%02d", z)) }
id <- sort(unique(canon(d$participant_id)))
if (length(id) != as.integer(args[3])) stop("Expected ", args[3], " participants; found ", length(id))
writeLines(id, args[2])
RS_IDS

make_bcb_manifest() {
  local branch="$1" dir="$2" output="$3"
  printf 'branch\tparticipant_id\tmap_path\n' > "$output"
  mapfile -t maps < <(find "$dir" -maxdepth 1 -type f \( -name 'sub-*_lesion.nii.gz' -o -name 'sub-*_lesion.nii' \) | sort)
  if [[ ${#maps[@]} -eq 0 ]]; then
    mapfile -t maps < <(find "$dir" -maxdepth 1 -type f \( -name 'sub-*.nii.gz' -o -name 'sub-*.nii' \) | sort)
  fi
  [[ ${#maps[@]} -eq "$EXPECTED_N" ]] || fail "$branch: expected ${EXPECTED_N} BCB maps in $dir, found ${#maps[@]}"
  local map base pid
  for map in "${maps[@]}"; do
    base="$(basename "$map")"
    pid="$(printf '%s\n' "$base" | grep -oE 'sub-[0-9]+' | head -1 || true)"
    [[ -n "$pid" ]] || fail "$branch: could not parse participant ID from $map"
    printf '%s\t%s\t%s\n' "$branch" "$pid" "$map" >> "$output"
  done
}
BCB_A_MANIFEST="${OUT}/audit/BCB_BranchA_map_manifest.tsv"
BCB_B_MANIFEST="${OUT}/audit/BCB_BranchB_map_manifest.tsv"
BCB_C_MANIFEST="${OUT}/audit/BCB_BranchC_map_manifest.tsv"
make_bcb_manifest BCB_BranchA "$BCB_A_DIR" "$BCB_A_MANIFEST"
make_bcb_manifest BCB_BranchB "$BCB_B_DIR" "$BCB_B_MANIFEST"
make_bcb_manifest BCB_BranchC "$BCB_C_DIR" "$BCB_C_MANIFEST"

Rscript --vanilla - "${OUT}/audit/expected_participants.txt" "$LNM_A_MANIFEST" "$LNM_B_MANIFEST" \
  "$BCB_A_MANIFEST" "$BCB_B_MANIFEST" "$BCB_C_MANIFEST" "${OUT}/audit/map_manifest_gate.tsv" "$EXPECTED_N" <<'RS_GATE'
args <- commandArgs(trailingOnly=TRUE)
expected <- readLines(args[1])
paths <- args[2:6]
labels <- c("LNM_BranchA","LNM_BranchB","BCB_BranchA","BCB_BranchB","BCB_BranchC")
expected_n <- as.integer(args[8])
rows <- list()
for (i in seq_along(paths)) {
  x <- read.delim(paths[i], stringsAsFactors=FALSE)
  required <- c("participant_id","map_path")
  if (!all(required %in% names(x))) stop("Manifest lacks required columns: ", paths[i])
  forbidden <- if (grepl("^LNM", labels[i])) any(grepl("lesions_weightedPV|lesionWeight|lesionSeed|voxelControlCount", x$map_path, ignore.case=TRUE)) else FALSE
  pass <- nrow(x)==expected_n && length(unique(x$participant_id))==expected_n && length(unique(x$map_path))==expected_n &&
          setequal(x$participant_id, expected) && all(file.exists(x$map_path)) && !forbidden
  rows[[i]] <- data.frame(branch=labels[i], n_rows=nrow(x), n_unique_ids=length(unique(x$participant_id)),
                          n_unique_paths=length(unique(x$map_path)), ids_match=setequal(x$participant_id, expected),
                          all_paths_exist=all(file.exists(x$map_path)), forbidden_LNM_paths=forbidden, pass=pass)
}
out <- do.call(rbind, rows)
write.table(out, args[7], sep="\t", quote=FALSE, row.names=FALSE)
if (!all(out$pass)) stop("Map manifest gate failed; inspect map_manifest_gate.tsv")
RS_GATE

printf 'branch\troi\tsource_mask\treference_map\tprepared_mask\n' > "${OUT}/audit/BCB_target_mask_manifest.tsv"
prepare_bcb_targets() {
  local branch="$1" manifest="$2" outdir="$3"
  local ref
  ref="$(awk -F'\t' 'NR==2 {print $3; exit}' "$manifest")"
  need_file "$ref"
  while IFS=$'\t' read -r roi label src; do
    [[ "$roi" == "roi" ]] && continue
    local raw="${OUT}/tmp/${branch}_${roi}_raw.nii.gz"
    local out="${outdir}/${roi}_on_${branch}_grid.nii.gz"
    flirt -in "$src" -ref "$ref" -applyxfm -usesqform -interp nearestneighbour -out "$raw"
    fslmaths "$raw" -thr 0.5 -bin "$out"
    rm -f "$raw"
    local nvox
    nvox="$(fslstats "$out" -V | awk '{print $1}')"
    [[ "$nvox" -gt 0 ]] || fail "$branch/$roi became empty after BCB-grid resampling"
    printf '%s\t%s\t%s\t%s\t%s\n' "$branch" "$roi" "$src" "$ref" "$out" >> "${OUT}/audit/BCB_target_mask_manifest.tsv"
  done < "${OUT}/audit/target_source_manifest.tsv"
}
prepare_bcb_targets BCB_BranchA "$BCB_A_MANIFEST" "${OUT}/masks/BCB_BranchA"
prepare_bcb_targets BCB_BranchB "$BCB_B_MANIFEST" "${OUT}/masks/BCB_BranchB"
prepare_bcb_targets BCB_BranchC "$BCB_C_MANIFEST" "${OUT}/masks/BCB_BranchC"

export OUT EXPECTED_N LNM_A_MANIFEST LNM_B_MANIFEST LNM_A_COVERAGE80 LNM_B_COVERAGE80
export BCB_A_MANIFEST BCB_B_MANIFEST BCB_C_MANIFEST
"$APPTAINER_BIN" exec "$FMRIPREP_SIF" python - <<'PY_EXTRACT'
import csv, math, os
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

OUT = Path(os.environ["OUT"])
EXPECTED_N = int(os.environ["EXPECTED_N"])

def fail(msg):
    raise RuntimeError(msg)

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def same_grid(a, b, atol=1e-4):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol, rtol=0)

def manifest(path, label):
    rows = read_rows(path)
    if len(rows) != EXPECTED_N:
        fail(f"{label}: expected {EXPECTED_N} maps, got {len(rows)}")
    out = []
    for r in rows:
        pid = r["participant_id"]
        p = Path(r["map_path"])
        if not p.is_file(): fail(f"{label}: map missing: {p}")
        out.append((pid, p))
    return sorted(out)

target_rows = read_rows(OUT / "audit/target_source_manifest.tsv")
targets = [(r["roi"], Path(r["source_mask"])) for r in target_rows]
expected = sorted([x.strip() for x in (OUT / "audit/expected_participants.txt").read_text().splitlines() if x.strip()])

lnm_cfg = [
    ("LNM_BranchA", Path(os.environ["LNM_A_MANIFEST"]), Path(os.environ["LNM_A_COVERAGE80"])),
    ("LNM_BranchB", Path(os.environ["LNM_B_MANIFEST"]), Path(os.environ["LNM_B_COVERAGE80"])),
]
bcb_cfg = [
    ("BCB_BranchA", Path(os.environ["BCB_A_MANIFEST"])),
    ("BCB_BranchB", Path(os.environ["BCB_B_MANIFEST"])),
    ("BCB_BranchC", Path(os.environ["BCB_C_MANIFEST"])),
]

raw_rows = []
mask_rows = []

for branch, man, coverage_path in lnm_cfg:
    maps = manifest(man, branch)
    if [p for p,_ in maps] != expected: fail(f"{branch}: participant IDs mismatch")
    reference = nib.load(str(maps[0][1]))
    coverage_img = nib.load(str(coverage_path))
    if not same_grid(reference, coverage_img): fail(f"{branch}: coverage/reference grid mismatch")
    coverage = coverage_img.get_fdata(dtype=np.float32) > 0.5
    prepared = {}
    for roi, source_path in targets:
        src = nib.load(str(source_path))
        rs = src if same_grid(src, reference) else resample_from_to(src, reference, order=1)
        weights = rs.get_fdata(dtype=np.float32)
        weights[~np.isfinite(weights)] = 0.0
        weights = np.clip(weights, 0.0, None)
        weights *= coverage.astype(np.float32)
        support = weights > 0.0001
        if int(support.sum()) < 1 or not math.isfinite(float(weights.sum())) or float(weights.sum()) <= 0:
            fail(f"{branch}/{roi}: empty weighted LNM target")
        out_mask = OUT / "masks" / branch / f"{roi}_weighted_on_{branch}_grid.nii.gz"
        hdr = reference.header.copy(); hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(weights.astype(np.float32), reference.affine, hdr), str(out_mask))
        prepared[roi] = (weights, support, out_mask)
        mask_rows.append(dict(modality="LNM", branch=branch, roi=roi, source_mask=str(source_path),
                              prepared_mask=str(out_mask), support_voxels=int(support.sum()),
                              weight_sum=float(weights.sum()), reference_map=str(maps[0][1]),
                              coverage80=str(coverage_path), preparation="order1_clip_nonnegative_times_coverage80_support_gt0.0001"))
    for pid, mp in maps:
        img = nib.load(str(mp))
        if not same_grid(img, reference): fail(f"{branch}/{pid}: Fisher-z map geometry mismatch")
        data = np.asanyarray(img.dataobj).astype(np.float32)
        for roi, (weights, support, out_mask) in prepared.items():
            vals = data[support]; w = weights[support]
            keep = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            if not keep.any(): fail(f"{branch}/{pid}/{roi}: no finite LNM voxels")
            mean_z = float(np.average(vals[keep], weights=w[keep]))
            if not math.isfinite(mean_z): fail(f"{branch}/{pid}/{roi}: non-finite LNM mean")
            raw_rows.append(dict(participant_id=pid, modality="LNM", branch=branch, roi=roi,
                                 estimand="mean_fisher_z", value=mean_z, map_path=str(mp),
                                 roi_mask=str(out_mask), roi_voxels=int(support.sum()), n_nonzero=""))

bcb_mask_manifest = read_rows(OUT / "audit/BCB_target_mask_manifest.tsv")
for branch, man in bcb_cfg:
    maps = manifest(man, branch)
    if [p for p,_ in maps] != expected: fail(f"{branch}: participant IDs mismatch")
    reference = nib.load(str(maps[0][1]))
    branch_masks = [r for r in bcb_mask_manifest if r["branch"] == branch]
    if len(branch_masks) != len(targets): fail(f"{branch}: target-mask manifest incomplete")
    prepared = {}
    for r in branch_masks:
        roi = r["roi"]
        img = nib.load(r["prepared_mask"])
        if not same_grid(img, reference): fail(f"{branch}/{roi}: prepared BCB mask grid mismatch")
        mask = img.get_fdata(dtype=np.float32) > 0.5
        if not mask.any(): fail(f"{branch}/{roi}: empty BCB mask")
        prepared[roi] = (mask, r["prepared_mask"], r["source_mask"])
        mask_rows.append(dict(modality="BCB", branch=branch, roi=roi, source_mask=r["source_mask"],
                              prepared_mask=r["prepared_mask"], support_voxels=int(mask.sum()),
                              weight_sum=int(mask.sum()), reference_map=str(maps[0][1]), coverage80="",
                              preparation="FSL_flirt_usesqform_nearestneighbour_thr0.5_bin"))
    for pid, mp in maps:
        img = nib.load(str(mp))
        if not same_grid(img, reference): fail(f"{branch}/{pid}: BCB map geometry mismatch")
        data = np.asanyarray(img.dataobj).astype(np.float64)
        finite_global = data[np.isfinite(data)]
        if finite_global.size == 0: fail(f"{branch}/{pid}: no finite BCB values")
        for roi, (mask, mask_path, source_path) in prepared.items():
            vals = data[mask]
            if vals.size != int(mask.sum()) or not np.isfinite(vals).all(): fail(f"{branch}/{pid}/{roi}: invalid ROI values")
            nz = vals[vals > 0]
            conditional = float(nz.mean()) if nz.size else 0.0
            mean_all = float(vals.mean())
            extent = float(100.0 * nz.size / vals.size)
            p95 = float(np.quantile(vals, .95))
            maximum = float(vals.max())
            raw_rows.append(dict(participant_id=pid, modality="BCB", branch=branch, roi=roi,
                                 estimand="conditional_nonzero_mean", value=conditional, map_path=str(mp),
                                 roi_mask=mask_path, roi_voxels=int(vals.size), n_nonzero=int(nz.size),
                                 mean_all=mean_all, extent_pct_nonzero=extent,
                                 integrated_probability_sum=float(vals.sum()), p95_all=p95, maximum=maximum))

write_rows(OUT / "audit/network_target_mask_manifest.tsv", mask_rows,
           ["modality","branch","roi","source_mask","prepared_mask","support_voxels","weight_sum","reference_map","coverage80","preparation"])
fields = ["participant_id","modality","branch","roi","estimand","value","map_path","roi_mask","roi_voxels","n_nonzero",
          "mean_all","extent_pct_nonzero","integrated_probability_sum","p95_all","maximum"]
write_rows(OUT / "tables/regional_subregion_predictors_raw_long.tsv", raw_rows, fields)

by_pid = {p:{"participant_id":p} for p in expected}
for r in raw_rows:
    key = f"REGION_{r['roi']}__{r['branch']}__{r['estimand']}"
    by_pid[r["participant_id"]][key] = r["value"]
all_keys = sorted({k for v in by_pid.values() for k in v if k != "participant_id"})
write_rows(OUT / "tables/regional_subregion_predictors_wide.tsv", [by_pid[p] for p in expected], ["participant_id"] + all_keys)

qc = []
for modality in ("LNM","BCB"):
  for branch in sorted({r["branch"] for r in raw_rows if r["modality"]==modality}):
    for roi,_ in targets:
      rr = [r for r in raw_rows if r["modality"]==modality and r["branch"]==branch and r["roi"]==roi]
      vals = np.array([float(r["value"]) for r in rr], float)
      n_nonzero = int(np.sum(vals != 0))
      qc.append(dict(modality=modality, branch=branch, roi=roi, n=len(vals), n_finite=int(np.isfinite(vals).sum()),
                     n_unique=int(np.unique(vals[np.isfinite(vals)]).size), n_nonzero=n_nonzero,
                     mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), min=float(np.min(vals)), max=float(np.max(vals)),
                     sparse_flag=(n_nonzero < 10 if modality=="BCB" else False),
                     usable=(len(vals)==EXPECTED_N and np.isfinite(vals).all() and np.std(vals,ddof=1)>1e-10 and np.unique(vals).size>=5)))
write_rows(OUT / "audit/predictor_distribution_QC.tsv", qc,
           ["modality","branch","roi","n","n_finite","n_unique","n_nonzero","mean","sd","min","max","sparse_flag","usable"])
if not all(r["usable"] for r in qc):
    bad = [f"{r['branch']}/{r['roi']}" for r in qc if not r["usable"]]
    fail("Unusable predictor(s): " + ", ".join(bad))
PY_EXTRACT

Rscript --vanilla - "$DATASET" "${OUT}/tables/regional_subregion_predictors_wide.tsv" "$OUT" <<'RS_ANALYSIS'
options(stringsAsFactors=FALSE, width=220, scipen=8)
args <- commandArgs(trailingOnly=TRUE)
DATASET <- args[1]; PRED <- args[2]; OUT <- args[3]

read_any <- function(path){
  first <- readLines(path,n=1,warn=FALSE); sep <- if(grepl("\t",first,fixed=TRUE)) "\t" else ","
  read.table(path,header=TRUE,sep=sep,quote="\"",comment.char="",check.names=FALSE,fill=TRUE,
             na.strings=c("","NA","NaN","n/a","N/A","null",".","#REF!","#VALUE!","#N/A"))
}
write_tsv <- function(x,name,subdir="tables"){
  p <- file.path(OUT,subdir,name); write.table(x,p,sep="\t",row.names=FALSE,quote=FALSE,na=""); p
}
num <- function(x) suppressWarnings(as.numeric(as.character(x)))
canon_pid <- function(x){ z <- suppressWarnings(as.integer(gsub("[^0-9]","",as.character(x)))); ifelse(is.na(z),as.character(x),sprintf("sub-%02d",z)) }
safe_z <- function(x){ x<-num(x); m<-mean(x,na.rm=TRUE); s<-sd(x,na.rm=TRUE); if(!is.finite(s)||s<=0) return(rep(NA_real_,length(x))); (x-m)/s }
safe_inverse <- function(M,tol=sqrt(.Machine$double.eps)){ direct<-tryCatch(solve(M),error=function(e)NULL); if(!is.null(direct))return(direct); z<-svd(M); cutoff<-max(z$d)*tol; dinv<-ifelse(z$d>cutoff,1/z$d,0); z$v %*% (dinv*t(z$u)) }
complete_model_data <- function(dat,vars){ vars<-unique(vars[!is.na(vars)&nzchar(vars)]); miss<-setdiff(vars,names(dat)); if(length(miss))stop("Missing variables: ",paste(miss,collapse=", ")); dat[complete.cases(dat[,vars,drop=FALSE]),vars,drop=FALSE] }
formula_string <- function(outcome,terms){ terms<-terms[!is.na(terms)&nzchar(terms)]; paste(outcome,"~",paste(terms,collapse=" + ")) }
bind_rows_base <- function(xs){ xs<-xs[!vapply(xs,is.null,logical(1))]; xs<-xs[vapply(xs,nrow,integer(1))>0]; if(!length(xs))return(data.frame()); cols<-unique(unlist(lapply(xs,names),use.names=FALSE)); filled<-lapply(xs,function(x){ for(m in setdiff(cols,names(x)))x[[m]]<-NA; x[,cols,drop=FALSE] }); do.call(rbind,filled) }

hc3_fit <- function(formula,dat,term,model_id=""){
  fit<-tryCatch(lm(formula,data=dat),error=function(e)NULL); if(is.null(fit))return(data.frame(model_id=model_id,term=term,status="FAILED_LM"))
  X<-model.matrix(fit); e<-residuals(fit); h<-hatvalues(fit); inv<-safe_inverse(crossprod(X)); omega<-e^2/pmax((1-h)^2,1e-12)
  vc<-inv %*% crossprod(X,X*omega) %*% inv; beta<-coef(fit); se<-sqrt(pmax(diag(vc),0)); df<-max(1,nrow(X)-qr(X)$rank); idx<-which(names(beta)==term)
  if(!length(idx))return(data.frame(model_id=model_id,term=term,status="TERM_NOT_ESTIMABLE")); tval<-beta[idx]/se[idx]; crit<-qt(.975,df)
  data.frame(model_id=model_id,model_type="OLS_HC3",n=nrow(X),events=NA,parameters=ncol(X),term=term,estimate=unname(beta[idx]),standard_error=unname(se[idx]),
             statistic=unname(tval),p_value=2*pt(abs(tval),df,lower.tail=FALSE),ci_low=unname(beta[idx]-crit*se[idx]),ci_high=unname(beta[idx]+crit*se[idx]),
             odds_ratio=NA,OR_ci_low=NA,OR_ci_high=NA,adjusted_R2=summary(fit)$adj.r.squared,AIC=AIC(fit),converged=TRUE,rank_deficient=qr(X)$rank<ncol(X),status="OK")
}
penloglik <- function(beta,X,y){ eta<-drop(X%*%beta); p<-plogis(pmax(pmin(eta,35),-35)); W<-pmax(p*(1-p),1e-10); I<-crossprod(X,X*W); detI<-determinant(I,logarithm=TRUE); if(detI$sign<=0)return(-Inf); sum(y*log(pmax(p,1e-15))+(1-y)*log(pmax(1-p,1e-15)))+.5*as.numeric(detI$modulus) }
firth_matrix <- function(X,y,maxit=300,tol=1e-8){
  b<-rep(0,ncol(X)); conv<-FALSE
  for(it in seq_len(maxit)){ eta<-drop(X%*%b); p<-plogis(pmax(pmin(eta,35),-35)); W<-pmax(p*(1-p),1e-10); I<-crossprod(X,X*W); inv<-safe_inverse(I); h<-rowSums((X%*%inv)*X)*W; score<-crossprod(X,y-p+h*(.5-p)); step<-drop(inv%*%score); old<-penloglik(b,X,y); fac<-1
    repeat{ cand<-b+fac*step; ll<-penloglik(cand,X,y); if(is.finite(ll)&&ll>=old-1e-10)break; fac<-fac/2; if(fac<1e-8)break }
    nb<-b+fac*step; if(max(abs(nb-b))<tol){b<-nb;conv<-TRUE;break}; b<-nb }
  eta<-drop(X%*%b); p<-plogis(pmax(pmin(eta,35),-35)); W<-pmax(p*(1-p),1e-10); I<-crossprod(X,X*W); vc<-safe_inverse(I); list(beta=b,vcov=vc,fitted=p,converged=conv,iterations=it)
}
firth_fit <- function(formula,dat,term,model_id=""){
  mf<-model.frame(formula,data=dat,na.action=na.omit); y<-model.response(mf); X<-model.matrix(formula,mf); if(length(unique(y))!=2L)return(data.frame(model_id=model_id,term=term,status="FAILED_BINARY_OUTCOME"))
  fit<-tryCatch(firth_matrix(X,y),error=function(e)NULL); idx<-match(term,colnames(X)); if(is.null(fit)||is.na(idx))return(data.frame(model_id=model_id,term=term,status="TERM_NOT_ESTIMABLE")); se<-sqrt(pmax(diag(fit$vcov),0)); z<-fit$beta[idx]/se[idx]
  data.frame(model_id=model_id,model_type="Firth_Jeffreys",n=nrow(X),events=sum(y==1),parameters=ncol(X),term=term,estimate=fit$beta[idx],standard_error=se[idx],statistic=z,p_value=2*pnorm(abs(z),lower.tail=FALSE),
             ci_low=fit$beta[idx]-1.96*se[idx],ci_high=fit$beta[idx]+1.96*se[idx],odds_ratio=exp(fit$beta[idx]),OR_ci_low=exp(fit$beta[idx]-1.96*se[idx]),OR_ci_high=exp(fit$beta[idx]+1.96*se[idx]),
             adjusted_R2=NA,AIC=NA,converged=fit$converged,rank_deficient=qr(X)$rank<ncol(X),status=ifelse(fit$converged,"OK","NONCONVERGED"))
}
huber_fit <- function(formula,dat,term,model_id="",maxit=100,tol=1e-7,k=1.345){
  mf<-model.frame(formula,data=dat,na.action=na.omit); y<-model.response(mf); X<-model.matrix(formula,mf); b<-tryCatch(qr.solve(X,y),error=function(e)rep(0,ncol(X))); conv<-FALSE
  for(it in seq_len(maxit)){ r<-y-drop(X%*%b); s<-median(abs(r-median(r)))/.6744898; if(!is.finite(s)||s<=1e-10)s<-sqrt(mean(r^2)); u<-r/pmax(s,1e-10); w<-ifelse(abs(u)<=k,1,k/abs(u)); Xw<-X*sqrt(w); yw<-y*sqrt(w); nb<-tryCatch(qr.solve(Xw,yw),error=function(e)b); if(max(abs(nb-b))<tol){b<-nb;conv<-TRUE;break}; b<-nb }
  idx<-match(term,colnames(X)); if(is.na(idx))return(data.frame(model_id=model_id,term=term,status="TERM_NOT_ESTIMABLE")); r<-y-drop(X%*%b); df<-max(1,nrow(X)-qr(X)$rank); sigma2<-sum(w*r^2)/df; vc<-sigma2*safe_inverse(crossprod(X,X*w)); se<-sqrt(pmax(diag(vc),0)); tval<-b[idx]/se[idx]
  data.frame(model_id=model_id,model_type="Huber_IRLS",n=nrow(X),term=term,estimate=b[idx],standard_error=se[idx],statistic=tval,p_value=2*pt(abs(tval),df,lower.tail=FALSE),ci_low=b[idx]-qt(.975,df)*se[idx],ci_high=b[idx]+qt(.975,df)*se[idx],converged=conv,status=ifelse(conv,"OK","MAXIT"))
}
fit_model <- function(dat,outcome,predictor,covars,model_id,binary=FALSE){
  d<-tryCatch(complete_model_data(dat,c(outcome,predictor,covars)),error=function(e)NULL); if(is.null(d)||nrow(d)<15L||length(unique(num(d[[predictor]])))<2L)return(data.frame(model_id=model_id,outcome=outcome,predictor=predictor,status="INSUFFICIENT_DATA"))
  d[[predictor]]<-num(d[[predictor]]); form<-as.formula(formula_string(outcome,c(predictor,covars))); out<-if(binary)firth_fit(form,d,predictor,model_id)else hc3_fit(form,d,predictor,model_id); out$outcome<-outcome; out$predictor<-predictor; out$covariates<-paste(covars,collapse="+"); out
}

main <- read_any(DATASET); pred <- read_any(PRED)
main$participant_id <- canon_pid(main$participant_id); pred$participant_id <- canon_pid(pred$participant_id)
if(nrow(main)!=63L||length(unique(main$participant_id))!=63L)stop("Final prepared dataset is not 63 unique participants")
if(nrow(pred)!=63L||length(unique(pred$participant_id))!=63L||!setequal(main$participant_id,pred$participant_id))stop("Predictor table participant gate failed")
dat <- merge(main,pred,by="participant_id",all.x=TRUE,sort=FALSE)

resolve_dat_col <- function(candidates, required=TRUE, label=paste(candidates,collapse=" / ")) {
  hit <- candidates[candidates %in% names(dat)]
  if(length(hit)) return(hit[1])
  if(required) stop("Could not resolve ", label, ". Tried: ", paste(candidates, collapse=", "))
  NA_character_
}

cpsp_src <- resolve_dat_col(c("CPSP_status_binary",".CPSP_binary","CPSP_binary"), TRUE, "CPSP status")
npsi_src <- resolve_dat_col(c("NPSI_total_four","NPSI_total","NPSI_total_model","NPSI"), TRUE, "NPSI total")
pc1_src  <- resolve_dat_col(c("QST_PC1_four","QST_PC1_saved","QST_PC1_model","QST_PC1","qst_pc1","PC1"), TRUE, "QST PC1")
pc2_src  <- resolve_dat_col(c("QST_PC2_four","QST_PC2_saved","QST_PC2_model","QST_PC2","qst_pc2","PC2"), TRUE, "saved QST PC2")
dat$CPSP_status_binary <- as.integer(num(dat[[cpsp_src]]) > 0)
dat$NPSI_total_four <- num(dat[[npsi_src]])
dat$QST_PC1_four <- num(dat[[pc1_src]])
dat$QST_PC2_four <- num(dat[[pc2_src]])

volz_src <- resolve_dat_col(c("z_log1p_lesion_volume","z_log1p_lesion_volume_four","z_reviewer_lesion_volume"), FALSE)
if(!is.na(volz_src)) {
  dat$z_log1p_lesion_volume <- num(dat[[volz_src]])
} else {
  logvol_src <- resolve_dat_col(c("log1p_lesion_volume"), FALSE)
  rawvol_src <- resolve_dat_col(c("lesion_volume_mm3_primary_BranchB","lesion_volume_mm3","lesion_volume"), FALSE)
  if(!is.na(logvol_src)) dat$z_log1p_lesion_volume <- safe_z(num(dat[[logvol_src]]))
  else if(!is.na(rawvol_src)) dat$z_log1p_lesion_volume <- safe_z(log1p(pmax(num(dat[[rawvol_src]]),0)))
  else stop("Could not resolve/derive lesion-volume covariate")
}

side_src <- resolve_dat_col(c("original_lesion_side_model","original_lesion_side","lesion_side"), TRUE, "lesion side")
dat$original_lesion_side_model <- factor(as.character(dat[[side_src]]))

age_src <- resolve_dat_col(c("z_age","z_age_four"), FALSE)
if(!is.na(age_src)) dat$z_age <- num(dat[[age_src]]) else {
  age_raw <- resolve_dat_col(c("age","Age","age_years"), FALSE)
  if(!is.na(age_raw)) dat$z_age <- safe_z(num(dat[[age_raw]]))
}
sex_src <- resolve_dat_col(c("sex","gender"), FALSE)
if(!is.na(sex_src)) dat$sex <- factor(as.character(dat[[sex_src]]))

chronic_src <- resolve_dat_col(c("z_log1p_months_since_stroke_four","z_log1p_months_since_stroke_manual","z_log1p_months_since_stroke","z_months_since_stroke"), FALSE)
if(!is.na(chronic_src)) {
  dat$z_log1p_months_since_stroke_four <- num(dat[[chronic_src]])
} else {
  months_src <- resolve_dat_col(c("months_since_stroke_manual","months_since_stroke","Months since stroke"), FALSE)
  if(!is.na(months_src)) dat$z_log1p_months_since_stroke_four <- safe_z(log1p(pmax(num(dat[[months_src]]),0)))
}

top5_src <- resolve_dat_col(c("sensitivity_exclude_large_lesion_top5","large_lesion_top5_flag"), FALSE)
if(!is.na(top5_src)) dat$large_lesion_top5_flag <- num(dat[[top5_src]])
top10_src <- resolve_dat_col(c("sensitivity_exclude_large_lesion_top10","large_lesion_top10_flag"), FALSE)
if(!is.na(top10_src)) dat$large_lesion_top10_flag <- num(dat[[top10_src]])

required_main <- c("CPSP_status_binary","NPSI_total_four","QST_PC1_four","QST_PC2_four","z_log1p_lesion_volume","original_lesion_side_model")
miss <- setdiff(required_main,names(dat)); if(length(miss))stop("Missing required main-model columns after canonicalisation: ",paste(miss,collapse=", "))

raw_preds <- grep("^REGION_",names(dat),value=TRUE)
for(p in raw_preds) dat[[paste0("z__",p)]] <- safe_z(dat[[p]])
write_tsv(dat,"analysis_dataset_with_regional_subregion_network_predictors.tsv")

roi_defs <- data.frame(
  roi=c("VPL","VPI","PuA","posterior_sensory_borderzone","opercular_cortex","insular_cortex","S2_OP1_OP4"),
  display=c("Morel VPL","Morel VPI","Morel PuA (anterior pulvinar)",
            "VMpo-related posterior sensory border-zone (VPLp+VPI+PuA; not atlas-defined VMpo)",
            "Opercular cortex (Harvard-Oxford Central + Parietal Operculum, 25% bilateral)",
            "Insular cortex (Harvard-Oxford Insular Cortex, 25% bilateral)",
            "S2 / parietal operculum (Juelich OP1-OP4, 25% bilateral)"),
  region_family=c("thalamus","thalamus","thalamus","thalamus",
                  "operculo_insular_S2","operculo_insular_S2","operculo_insular_S2"),
  stringsAsFactors=FALSE
)
outcomes <- data.frame(
  outcome=c("CPSP_status_binary","NPSI_total_four","QST_PC1_four","QST_PC2_four"),
  outcome_label=c("CPSP_status","NPSI_total","QST_PC1","QST_PC2"),
  binary=c(TRUE,FALSE,FALSE,FALSE), stringsAsFactors=FALSE
)
base_covars <- c("z_log1p_lesion_volume","original_lesion_side_model")
branch_specs <- data.frame(
  branch=c("LNM_BranchB","BCB_BranchB","LNM_BranchA","BCB_BranchA","BCB_BranchC"),
  modality=c("LNM","BCB","LNM","BCB","BCB"),
  public_branch=c("Main branch","Main branch","Sensitivity Branch B","Sensitivity Branch B","Sensitivity Branch C"),
  estimand=c("mean_fisher_z","conditional_nonzero_mean","mean_fisher_z","conditional_nonzero_mean","conditional_nonzero_mean"),
  is_main=c(TRUE,TRUE,FALSE,FALSE,FALSE), stringsAsFactors=FALSE
)
predictor_name <- function(roi,branch,estimand) paste0("z__REGION_",roi,"__",branch,"__",estimand)

run_family <- function(dat_use,specs,covars,adjustment="minimal",subset_label="full"){
  rows<-list(); k<-0L
  for(i in seq_len(nrow(specs))) for(j in seq_len(nrow(roi_defs))) for(h in seq_len(nrow(outcomes))){
    sp<-specs[i,]; rr<-roi_defs[j,]; oo<-outcomes[h,]; pcol<-predictor_name(rr$roi,sp$branch,sp$estimand)
    if(!pcol%in%names(dat_use))stop("Predictor missing: ",pcol)
    k<-k+1L; mid<-paste("regional_localisation",rr$roi,sp$branch,oo$outcome_label,adjustment,subset_label,sep="__")
    z<-fit_model(dat_use,oo$outcome,pcol,covars,mid,oo$binary)
    z$roi<-rr$roi; z$roi_display<-rr$display; z$region_family<-rr$region_family; z$modality<-sp$modality; z$branch_internal<-sp$branch; z$branch_label<-sp$public_branch; z$estimand<-sp$estimand; z$outcome_label<-oo$outcome_label; z$adjustment<-adjustment; z$subset<-subset_label
    rows[[k]]<-z
  }
  bind_rows_base(rows)
}
apply_main_fdr <- function(x){
  x$q_within_target_modality_4 <- ave(num(x$p_value),interaction(x$roi,x$modality,drop=TRUE),FUN=function(p)p.adjust(p,"BH"))
  x$q_within_family_modality <- ave(num(x$p_value),interaction(x$region_family,x$modality,drop=TRUE),FUN=function(p)p.adjust(p,"BH"))
  x$q_within_modality_28 <- ave(num(x$p_value),x$modality,FUN=function(p)p.adjust(p,"BH"))
  x$q_within_family_all_modalities <- ave(num(x$p_value),x$region_family,FUN=function(p)p.adjust(p,"BH"))
  x$q_all_56 <- p.adjust(num(x$p_value),"BH")
  x
}

main_specs <- branch_specs[branch_specs$is_main,,drop=FALSE]
main_res <- run_family(dat,main_specs,base_covars)
if(nrow(main_res)!=56L)stop("Expected 56 Main localisation models; got ",nrow(main_res))
main_res <- apply_main_fdr(main_res)
write_tsv(main_res,"regional_subregion_main_four_outcome_results.tsv")

sens_specs <- branch_specs[!branch_specs$is_main,,drop=FALSE]
sens <- run_family(dat,sens_specs,base_covars,adjustment="branch_sensitivity")
sens$q_within_target_branch_4 <- ave(num(sens$p_value),interaction(sens$roi,sens$branch_internal,drop=TRUE),FUN=function(p)p.adjust(p,"BH"))
sens$q_within_branch_family <- ave(num(sens$p_value),interaction(sens$branch_internal,sens$region_family,drop=TRUE),FUN=function(p)p.adjust(p,"BH"))
sens$q_within_branch_28 <- ave(num(sens$p_value),sens$branch_internal,FUN=function(p)p.adjust(p,"BH"))
write_tsv(sens,"regional_subregion_branch_sensitivities.tsv")

adj_specs <- list()
if(all(c("z_age","sex") %in% names(dat))) adj_specs$age_sex <- c(base_covars,"z_age","sex")
if("z_log1p_months_since_stroke_four" %in% names(dat)) adj_specs$chronicity <- c(base_covars,"z_log1p_months_since_stroke_four")
if(all(c("z_age","sex","z_log1p_months_since_stroke_four") %in% names(dat))) adj_specs$age_sex_chronicity <- c(base_covars,"z_age","sex","z_log1p_months_since_stroke_four")
adj_rows<-list(); for(nm in names(adj_specs)){ z<-run_family(dat,main_specs,adj_specs[[nm]],adjustment=nm); z<-apply_main_fdr(z); adj_rows[[nm]]<-z }
adj <- bind_rows_base(adj_rows); write_tsv(adj,"regional_subregion_age_sex_chronicity_sensitivities.tsv")

exc_rows<-list()
for(nm in c("exclude_top5","exclude_top10")){
  flag <- if(nm=="exclude_top5") "large_lesion_top5_flag" else "large_lesion_top10_flag"
  if(!flag %in% names(dat)) next
  keep <- is.na(num(dat[[flag]])) | num(dat[[flag]])==0
  z<-run_family(dat[keep,,drop=FALSE],main_specs,base_covars,adjustment="minimal",subset_label=nm); z<-apply_main_fdr(z); z$excluded_n<-sum(!keep); exc_rows[[nm]]<-z
}
exc<-bind_rows_base(exc_rows); write_tsv(exc,"regional_subregion_large_lesion_exclusions.tsv")

huber_rows<-list(); hk<-0L
cont <- main_res[!main_res$outcome_label%in%"CPSP_status",,drop=FALSE]
for(i in seq_len(nrow(cont))){ m<-cont[i,]; d<-complete_model_data(dat,c(m$outcome,m$predictor,base_covars)); form<-as.formula(formula_string(m$outcome,c(m$predictor,base_covars))); r<-huber_fit(form,d,m$predictor,m$model_id); r$roi<-m$roi; r$roi_display<-m$roi_display; r$region_family<-m$region_family; r$modality<-m$modality; r$outcome_label<-m$outcome_label; hk<-hk+1L; huber_rows[[hk]]<-r }
huber<-bind_rows_base(huber_rows); huber$q_within_modality_21<-ave(num(huber$p_value),huber$modality,FUN=function(p)p.adjust(p,"BH")); huber$q_all_42<-p.adjust(num(huber$p_value),"BH"); write_tsv(huber,"regional_subregion_Huber_robust_regression.tsv")

inf_rows<-list(); ik<-0L
for(i in seq_len(nrow(cont))){ m<-cont[i,]; d<-complete_model_data(dat,c("participant_id",m$outcome,m$predictor,base_covars)); fit<-lm(as.formula(formula_string(m$outcome,c(m$predictor,base_covars))),data=d); db<-dfbeta(fit); dbp<-if(m$predictor%in%colnames(db))db[,m$predictor]else rep(NA_real_,nrow(d)); X<-model.matrix(fit)
  z<-data.frame(model_id=m$model_id,roi=m$roi,roi_display=m$roi_display,region_family=m$region_family,modality=m$modality,outcome_label=m$outcome_label,participant_id=d$participant_id,cooks_distance=cooks.distance(fit),leverage=hatvalues(fit),studentised_residual=rstudent(fit),dfbeta_predictor=dbp,stringsAsFactors=FALSE)
  z$flag_cook<-z$cooks_distance>4/nrow(d); z$flag_leverage<-z$leverage>2*ncol(X)/nrow(d); z$flag_studentised<-abs(z$studentised_residual)>3; z$flag_dfbeta<-abs(z$dfbeta_predictor)>2/sqrt(nrow(d)); z$any_influence_flag<-z$flag_cook|z$flag_leverage|z$flag_studentised|z$flag_dfbeta
  ik<-ik+1L; inf_rows[[ik]]<-z
}
inf<-bind_rows_base(inf_rows); write_tsv(inf,"regional_subregion_influence_diagnostics_long.tsv")
summary_rows<-do.call(rbind,lapply(split(inf,inf$model_id),function(x)data.frame(model_id=x$model_id[1],roi=x$roi[1],region_family=x$region_family[1],modality=x$modality[1],outcome_label=x$outcome_label[1],n=nrow(x),n_any_flag=sum(x$any_influence_flag,na.rm=TRUE),max_cooks=max(x$cooks_distance,na.rm=TRUE),max_leverage=max(x$leverage,na.rm=TRUE),max_abs_studentised=max(abs(x$studentised_residual),na.rm=TRUE),max_abs_dfbeta=max(abs(x$dfbeta_predictor),na.rm=TRUE))))
write_tsv(summary_rows,"regional_subregion_influence_diagnostics_summary.tsv")

pcols <- grep("^z__REGION_.*__(LNM_BranchB|BCB_BranchB)__",names(dat),value=TRUE)
desc<-do.call(rbind,lapply(pcols,function(p)data.frame(predictor=p,n_finite=sum(is.finite(num(dat[[p]]))),mean=mean(num(dat[[p]]),na.rm=TRUE),sd=sd(num(dat[[p]]),na.rm=TRUE),min=min(num(dat[[p]]),na.rm=TRUE),max=max(num(dat[[p]]),na.rm=TRUE))))
write_tsv(desc,"regional_subregion_standardised_predictor_descriptives.tsv")
for(mod in c("LNM","BCB")){
  cols<-grep(paste0("^z__REGION_.*__",mod,"_BranchB__"),names(dat),value=TRUE)
  cm<-cor(dat[,cols,drop=FALSE],use="pairwise.complete.obs"); write.table(cm,file.path(OUT,"tables",paste0("regional_subregion_",mod,"_predictor_correlations.tsv")),sep="\t",quote=FALSE,col.names=NA)
}

headline <- main_res[,c("region_family","roi_display","modality","outcome_label","model_type","n","events","estimate","ci_low","ci_high","odds_ratio","OR_ci_low","OR_ci_high","p_value","q_within_target_modality_4","q_within_family_modality","q_within_modality_28","q_within_family_all_modalities","q_all_56","status")]
write_tsv(headline,"REGIONAL_SUBREGION_HEADLINE_TABLE.tsv")

cat("Main rows:",nrow(main_res),"\n")
cat("Branch sensitivity rows:",nrow(sens),"\n")
cat("Covariate sensitivity rows:",nrow(adj),"\n")
cat("Large-lesion exclusion rows:",nrow(exc),"\n")
cat("Huber rows:",nrow(huber),"\n")
cat("Influence rows:",nrow(inf),"\n")
RS_ANALYSIS

need_file "${OUT}/tables/regional_subregion_main_four_outcome_results.tsv"
need_file "${OUT}/tables/REGIONAL_SUBREGION_HEADLINE_TABLE.tsv"
Rscript --vanilla - "${OUT}/tables/regional_subregion_main_four_outcome_results.tsv" <<'RS_FINAL_GATE'
a <- commandArgs(trailingOnly=TRUE)
x <- read.delim(a[1],stringsAsFactors=FALSE)
if(nrow(x)!=56L)stop("Final gate: expected 56 Main rows; found ",nrow(x))
if(any(!x$status %in% c("OK","NONCONVERGED")))stop("Final gate: failed/insufficient Main models present")
if(sum(x$modality=="LNM")!=28L||sum(x$modality=="BCB")!=28L)stop("Final gate: modality row counts incorrect")
cat("Final model gate PASS\n")
RS_FINAL_GATE

cat > "${OUT}/report/README.txt" <<EOF
Thalamic + operculo-insular/S2 subregion LNM/BCB four-outcome localisation analysis completed.

Primary table:
  ${OUT}/tables/REGIONAL_SUBREGION_HEADLINE_TABLE.tsv
Full Main table:
  ${OUT}/tables/regional_subregion_main_four_outcome_results.tsv

Targets:
  Thalamus family (4): Morel VPL, VPI, PuA, posterior sensory border-zone (VPLp+VPI+PuA).
  Operculo-insular/S2 family (3): Harvard-Oxford opercular cortex, Harvard-Oxford insular cortex,
  and Juelich S2/parietal operculum OP1-OP4, all bilateral and thresholded at 25% for source-mask construction.

Main correction structure:
  q_within_target_modality_4: 4 outcomes within each ROI x modality.
  q_within_family_modality: 16 thalamic tests or 12 operculo-insular/S2 tests within each modality.
  q_within_modality_28: 28 tests within LNM and 28 within BCB.
  q_within_family_all_modalities: 32 thalamic tests or 24 operculo-insular/S2 tests across modalities.
  q_all_56: all 56 Main localisation tests.

Interpretation notes:
  The posterior sensory border-zone is VPLp+VPI+PuA and is NOT an atlas-defined VMpo parcel.
  Opercular cortex and Juelich S2 overlap anatomically by design; these are localisation targets,
  not independent replication tests. The three cortical targets are subdivisions/localisations of
  the previously analysed broad operculo-insular/S2 system.
  This script does not rerun PALM/randomise/permutations and does not alter locked prior results.
EOF

cat > "${OUT}/SUCCESS.flag" <<EOF
SUCCESS
completed=$(date --iso-8601=seconds)
main_models=56
main_LNM_models=28
main_BCB_models=28
thalamus_main_models=32
operculo_insular_S2_main_models=24
permutations=NOT_RUN
EOF

echo "============================================================"
echo "THALAMIC + OPERCULO-INSULAR/S2 SUBREGION FOUR-OUTCOME ANALYSIS COMPLETE"
echo "OUT: ${OUT}"
echo "Headline: ${OUT}/tables/REGIONAL_SUBREGION_HEADLINE_TABLE.tsv"
echo "============================================================"
SOURCE_RUN_THALAMIC_OPERCULO_INSULAR_S2_SUBREGIONS_LNM_BCB_FOUR_OUTCOMES_V1_1_FIXED_SH

cat > "$WORK/prepare_partB_randomise_bulletproof_v1_1.sh" <<'SOURCE_PREPARE_PARTB_RANDOMISE_BULLETPROOF_V1_1_SH'
#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "ERROR: do not source this script." >&2
    return 2
fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
PARTB="${PARTB:-${RUN}/MASTER_REBUILT_PART_B_20260710_215017}"
ATOG="${ATOG:-${RUN}/AtoG_spatial_followups_20260707_214137}"
DATASET="${DATASET:-${PARTB}/tables/model_dataset_REBUILT_structured_HCPex_LNM.tsv}"

BRANCH_A_FISHERZ_DIR="${BRANCH_A_FISHERZ_DIR:-${PROJECT}/derivatives/phase3_stroke/functional_lnm_v3_fmriprepPV_sensitivity/lnm_maps/HC19_exclSub09Sub17}"
BRANCH_B_FISHERZ_DIR="${BRANCH_B_FISHERZ_DIR:-${PROJECT}/derivatives/phase3_stroke/functional_lnm_v3_enantiomorphicPV/lnm_maps/HC19_exclSub09Sub17}"

ADD_LNM_TOP10="${ADD_LNM_TOP10:-1}"

V3_AUDIT="${V3_AUDIT:-}"
if [[ -z "$V3_AUDIT" ]]; then
    V3_AUDIT="$(find "$RUN" -maxdepth 1 -type d -name 'RANDOMISE_STACK_DESIGN_AUDIT_V3_*' -exec test -f '{}/SUCCESS.flag' ';' -print | sort | tail -1)"
fi

OUT="${OUT:-${RUN}/BULLETPROOF_RANDOMISE_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"/{audit,designs,legacy_quarantine,logs,manifests,masks,report,scripts,stacks,tables,tmp}
LOG="$OUT/logs/prepare.log"
exec > >(tee "$LOG") 2>&1

on_error() {
    rc=$?
    trap - ERR
    {
        echo "FAILED"
        echo "stage=prepare"
        echo "exit_code=$rc"
        echo "time=$(date --iso-8601=seconds 2>/dev/null || date)"
        echo "line=${BASH_LINENO[0]:-unknown}"
        echo "command=${BASH_COMMAND:-unknown}"
        echo "log=$LOG"
    } > "$OUT/PREPARE_FAILED.flag"
    echo "ERROR: preparation failed. See $OUT/PREPARE_FAILED.flag and $LOG"
    exit "$rc"
}
trap on_error ERR

for path in "$DATASET" "$BRANCH_A_FISHERZ_DIR" "$BRANCH_B_FISHERZ_DIR" "$ATOG/stacks" "$ATOG/designs" "$ATOG/randomise" "$V3_AUDIT/SUCCESS.flag" "$V3_AUDIT/tables/randomise_volume_participant_design_matches.tsv" "$V3_AUDIT/tables/randomise_models_certified_existing_outputs.tsv"; do
    [[ -e "$path" ]] || { echo "ERROR: required input missing: $path" >&2; exit 1; }
done

module purge >/dev/null 2>&1 || true
module load apptainer >/dev/null 2>&1 || true
APPTAINER_BIN="${APPTAINER_BIN:-/opt/apps/pkg/tools/apptainer/1.3.6/bin/apptainer}"
[[ -x "$APPTAINER_BIN" ]] || APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
FMRIPREP_SIF="${FMRIPREP_SIF:-/opt/apps/pkg/applications/containers/fmriprep/25.1.3/fmriprep_25.1.3.sif}"
[[ -n "$APPTAINER_BIN" && -x "$APPTAINER_BIN" ]] || { echo "ERROR: apptainer unavailable" >&2; exit 1; }
[[ -f "$FMRIPREP_SIF" ]] || { echo "ERROR: container unavailable: $FMRIPREP_SIF" >&2; exit 1; }

cp -p "${BASH_SOURCE[0]}" "$OUT/scripts/$(basename "${BASH_SOURCE[0]}")"

{
    echo "version=2026-07-12.bulletproof-v1.1-authoritative-FisherZ-roots"
    echo "PROJECT=$PROJECT"
    echo "RUN=$RUN"
    echo "PARTB=$PARTB"
    echo "ATOG=$ATOG"
    echo "DATASET=$DATASET"
    echo "BRANCH_A_FISHERZ_DIR=$BRANCH_A_FISHERZ_DIR"
    echo "BRANCH_B_FISHERZ_DIR=$BRANCH_B_FISHERZ_DIR"
    echo "V3_AUDIT=$V3_AUDIT"
    echo "ADD_LNM_TOP10=$ADD_LNM_TOP10"
    echo "OUT=$OUT"
} > "$OUT/audit/provenance.txt"

find "$PROJECT/derivatives/phase3_stroke" -type f \( -name '*.nii' -o -name '*.nii.gz' \) 2>/dev/null \
    | grep -Ei '/functional_lnm|/lnm|lesion.network|connectivity' \
    | grep -Eiv '/AtoG_spatial_followups_|/randomise/|_4D\.nii|group_mean|groupmean|tfce|tstat|fstat|corrp' \
    > "$OUT/tmp/all_lnm_candidate_paths.txt" || true

export PROJECT RUN PARTB ATOG DATASET V3_AUDIT OUT ADD_LNM_TOP10
export BRANCH_A_FISHERZ_DIR BRANCH_B_FISHERZ_DIR

"$APPTAINER_BIN" exec \
    -B /mnt/scratch:/mnt/scratch \
    -B /opt/apps:/opt/apps \
    "$FMRIPREP_SIF" python - <<'PY'
import csv, hashlib, math, os, re, shutil
from collections import defaultdict
from pathlib import Path
import nibabel as nib
import numpy as np

PROJECT=Path(os.environ['PROJECT']); RUN=Path(os.environ['RUN']); PARTB=Path(os.environ['PARTB'])
ATOG=Path(os.environ['ATOG']); DATASET=Path(os.environ['DATASET']); V3=Path(os.environ['V3_AUDIT']); OUT=Path(os.environ['OUT'])
BRANCH_A_FISHERZ_DIR=Path(os.environ['BRANCH_A_FISHERZ_DIR'])
BRANCH_B_FISHERZ_DIR=Path(os.environ['BRANCH_B_FISHERZ_DIR'])
ADD_TOP10=os.environ.get('ADD_LNM_TOP10','1')=='1'

BRANCH_TOKENS={
 'LNM_BranchA': ('brancha','fmripreppv','fmriprep'),
 'LNM_BranchB': ('branchb','enantiomorphic','weightedpv'),
}
VALID_MAP_TOKENS=('fisherz','fisher_z','fisher-z','connectivitymap','connectivity_map','seedfc','fcmap','correlationmap','corrmap','zmap')
INVALID_MAP_TOKENS=('lesionweight','lesion_weight','weightedpv_lesion','lesions_weightedpv','lesionseed','coverage','controlcount','meanmap','groupmean','group_mean','stdmap','variance','mask','roi')

def read_table(path):
    lines=path.read_text(errors='replace').splitlines()
    if not lines: return []
    sep='\t' if '\t' in lines[0] else ','
    with path.open(newline='') as h: return list(csv.DictReader(h,delimiter=sep))

def write_tsv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else [])
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def norm(x): return re.sub(r'[^a-z0-9]','',str(x).strip().lower())
def pid(x):
    m=re.search(r'sub-[A-Za-z0-9]+',str(x)); return m.group(0) if m else str(x).strip()
def col(rows,aliases):
    if not rows:return None
    d={norm(k):k for k in rows[0]}
    for a in aliases:
        if norm(a) in d:return d[norm(a)]
    return None

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def parse_vest(path):
    lines=path.read_text(errors='replace').splitlines(); head={}; start=None
    for i,line in enumerate(lines):
        s=line.strip()
        if s.startswith('/') and ' ' in s:
            k,v=s.split(None,1);head[k]=v.strip()
        if s=='/Matrix':start=i+1;break
    if start is None:raise RuntimeError(f'No /Matrix: {path}')
    mat=np.asarray([[float(x) for x in line.split()] for line in lines[start:] if line.strip()],float)
    return head,mat

def read_design_columns(path,n):
    rows=read_table(path)
    idx=col(rows,('column_index','index','wave','column'))
    name=col(rows,('column_name','design_column','name','variable','term'))
    if name:
        ordered=sorted(rows,key=lambda r:int(float(r[idx]))) if idx else rows
        names=[str(r[name]).strip() for r in ordered]
    elif rows and len(rows[0])==1:
        k=next(iter(rows[0]));names=[str(r[k]).strip() for r in rows]
    else: raise RuntimeError(f'Cannot parse design columns: {path}')
    if len(names)!=n:raise RuntimeError(f'Design column count mismatch: {path}')
    return names

def canonical_level(x):
    s=norm(x);aliases={'nnpsp':'nn','nonneuropathic':'nn','nonneuropathicpain':'nn','centralpoststrokepain':'cpsp','strokenopain':'snp','nopain':'snp','m':'male','f':'female','l':'left','r':'right','mixed':'bilateral','bilateralormixed':'bilateral'}
    return aliases.get(s,s)

def numeric(x):
    try:return float(str(x).strip())
    except:return float('nan')

metadata=read_table(DATASET); mp=col(metadata,('participant_id','subject_id')); mg=col(metadata,('group','analysis_group'))
if not mp:raise RuntimeError('Dataset lacks participant_id')
meta={pid(r[mp]):r for r in metadata}; meta_cols={norm(k):k for k in metadata[0]}

def design_value(row,name):
    key=norm(name)
    if key in ('intercept','constant','const'):return 1.0
    if key in meta_cols:
        v=numeric(row.get(meta_cols[key],''))
        if math.isfinite(v):return v
    g=canonical_level(row.get(mg,'')) if mg else ''
    targets={'cpsp':'cpsp','groupcpsp':'cpsp','nn':'nn','groupnn':'nn','nnpsp':'nn','groupnnpsp':'nn','snp':'snp','groupsnp':'snp'}
    if key in targets:return float(g==targets[key])
    matches=[]
    for base,bcol in meta_cols.items():
        if base and key.startswith(base) and key!=base:
            suffix=key[len(base):]
            if suffix:matches.append((len(base),float(canonical_level(row.get(bcol,''))==canonical_level(suffix))))
    if matches:return sorted(matches,reverse=True)[0][1]
    compact={'sexmale':('sex','male'),'sexfemale':('sex','female'),'male':('sex','male'),'female':('sex','female'),'originallesionsidemodelright':('original_lesion_side_model','right'),'originallesionsidemodelleft':('original_lesion_side_model','left'),'originallesionsidemodelbilateral':('original_lesion_side_model','bilateral')}
    if key in compact:
        base,level=compact[key]; bcol=meta_cols.get(norm(base))
        if bcol:return float(canonical_level(row.get(bcol,''))==level)
    raise RuntimeError(f'Cannot reconstruct design column {name}')

def expected_vector(participant,names):
    if participant not in meta:raise RuntimeError(f'Missing metadata: {participant}')
    return np.asarray([design_value(meta[participant],x) for x in names],float)

vol=read_table(V3/'tables'/'randomise_volume_participant_design_matches.tsv')
vp=col(vol,('matched_participant_id','participant_id')); vm=col(vol,('modality',)); vd=col(vol,('design_name',)); vi=col(vol,('volume_index_1based',))
orders=defaultdict(list)
for r in vol:
    if not r.get(vp):continue
    orders[(r[vm],r[vd])].append((int(float(r[vi])),pid(r[vp])))
orders={k:[p for _,p in sorted(v)] for k,v in orders.items()}

specs=[]
for branch in ('LNM_BranchA','LNM_BranchB'):
    for design,roles in (
      ('stroke_all_minimal','full_sample_primary_spatial_followup'),
      ('stroke_all_age_sex_adjusted','age_sex_sensitivity'),
      ('strict_nonCPSP_exclude_NN_DN4pos_minimal','strict_control_sensitivity'),
      ('exclude_low_LNM_coverage_minimal','low_coverage_sensitivity'),
    ):
        specs.append((branch,design,roles))
    if ADD_TOP10:specs.append((branch,'exclude_large_top10_minimal','lesion_size_sensitivity'))

model_orders={}
for branch,design,role in specs:
    key=(branch,design)
    if key in orders:model_orders[key]=orders[key]
    elif design=='exclude_large_top10_minimal' and ('BCB_BranchB',design) in orders:model_orders[key]=orders[('BCB_BranchB',design)]
    else:raise RuntimeError(f'No authoritative participant order for {branch}/{design}')

AUTHORITATIVE_ROOTS = {
    'LNM_BranchA': BRANCH_A_FISHERZ_DIR,
    'LNM_BranchB': BRANCH_B_FISHERZ_DIR,
}

EXPECTED_PATH_TOKENS = {
    'LNM_BranchA': (
        'functional_lnm_v3_fmriprepPV_sensitivity',
        'functionalLNMv3WeightedPV_branchA_fmriprep2mm_sensitivity',
        'HC19_exclSub09Sub17',
        'fisherZ',
    ),
    'LNM_BranchB': (
        'functional_lnm_v3_enantiomorphicPV',
        'functionalLNMv3WeightedPV_branchB_enantiomorphic2mm_QCgated',
        'HC19_exclSub09Sub17',
        'fisherZ',
    ),
}

all_participants = sorted({
    participant
    for order in model_orders.values()
    for participant in order
})

candidate_audit = []
selected = {}
root_audit = []

for branch, root in AUTHORITATIVE_ROOTS.items():
    if not root.is_dir():
        raise RuntimeError(
            f'Authoritative Fisher-Z directory missing for {branch}: {root}'
        )

    files = sorted(root.glob('sub-*_fisherZ.nii.gz'))

    if not files:
        raise RuntimeError(
            f'No Fisher-Z maps found in authoritative directory for {branch}: {root}'
        )

    files_by_participant = defaultdict(list)

    for path in files:
        participant = pid(path.name)
        files_by_participant[participant].append(path)

    duplicate_participants = {
        participant: paths
        for participant, paths in files_by_participant.items()
        if len(paths) != 1
    }

    if duplicate_participants:
        details = '; '.join(
            f'{participant}={len(paths)}'
            for participant, paths in sorted(duplicate_participants.items())
        )
        raise RuntimeError(
            f'Duplicate Fisher-Z maps within authoritative {branch} root: {details}'
        )

    available_participants = set(files_by_participant)
    required_participants = set(all_participants)
    missing = sorted(required_participants - available_participants)

    if missing:
        raise RuntimeError(
            f'Authoritative {branch} Fisher-Z root is missing '
            f'{len(missing)} required participants: {missing}'
        )

    unexpected = sorted(available_participants - set(meta))

    root_audit.append({
        'branch': branch,
        'authoritative_root': str(root),
        'n_fisherz_files': len(files),
        'n_unique_participants': len(available_participants),
        'n_required_participants': len(required_participants),
        'n_required_missing': len(missing),
        'n_participants_not_in_authoritative_dataset': len(unexpected),
        'legacy_v2_directory_permitted': 0,
        'pass': 1,
    })

    for participant in all_participants:
        path = files_by_participant[participant][0]
        path_text = str(path)

        missing_tokens = [
            token
            for token in EXPECTED_PATH_TOKENS[branch]
            if token not in path_text
        ]

        if missing_tokens:
            raise RuntimeError(
                f'Authoritative path-token audit failed for '
                f'{branch}/{participant}: missing={missing_tokens}; path={path}'
            )

        if 'functional_lnm_v2_weightedPV' in path_text:
            raise RuntimeError(
                f'Legacy v2 Fisher-Z map entered authoritative selection: {path}'
            )

        selected[(branch, participant)] = path

        candidate_audit.append({
            'branch': branch,
            'participant_id': participant,
            'rank': 1,
            'score': 'AUTHORITATIVE_EXACT_ROOT',
            'path': str(path),
            'selected': 1,
            'selection_rule': (
                'exact participant match within frozen authoritative v3 directory'
            ),
            'legacy_v2_excluded': 1,
        })

write_tsv(
    OUT/'tables'/'FisherZ_authoritative_root_audit.tsv',
    root_audit,
)

write_tsv(
    OUT/'tables'/'FisherZ_candidate_resolution_audit.tsv',
    candidate_audit,
)

map_cache={}; map_manifest=[]; branch_geometry={}
for (branch,participant),path in sorted(selected.items()):
    img=nib.load(str(path)); data=np.asanyarray(img.dataobj)
    data=np.squeeze(data)
    if data.ndim!=3:raise RuntimeError(f'Fisher-Z map not 3D: {path}')
    if not np.all(np.isfinite(data)):raise RuntimeError(f'Non-finite Fisher-Z values: {path}')
    geometry=(data.shape,tuple(np.round(img.affine.ravel(),6)))
    branch_geometry.setdefault(branch,geometry)
    if geometry!=branch_geometry[branch]:raise RuntimeError(f'Geometry mismatch within {branch}: {path}')
    if np.nanstd(data)<1e-8:raise RuntimeError(f'Near-constant Fisher-Z map: {path}')
    map_cache[(branch,participant)]=(img,np.asarray(data,dtype=np.float32))
    map_manifest.append({'branch':branch,'participant_id':participant,'fisherz_map_path':str(path),'sha256':sha(path),'shape':'x'.join(map(str,data.shape)),'voxel_sizes_mm':';'.join(map(str,np.round(img.header.get_zooms()[:3],6))),'minimum':float(np.min(data)),'maximum':float(np.max(data)),'mean':float(np.mean(data)),'sd':float(np.std(data))})
write_tsv(OUT/'manifests'/'FisherZ_participant_map_manifest.tsv',map_manifest)

branch_masks={}
for branch in ('LNM_BranchA','LNM_BranchB'):
    participants=sorted(p for b,p in selected if b==branch)
    arrays=[map_cache[(branch,p)][1] for p in participants]
    stack=np.stack(arrays,axis=3)
    finite=np.all(np.isfinite(stack),axis=3)
    informative=np.any(np.abs(stack)>1e-8,axis=3)
    variance=np.std(stack,axis=3)>1e-8
    mask=finite & informative & variance
    if int(mask.sum())<1000:raise RuntimeError(f'Implausibly small common mask for {branch}: {int(mask.sum())}')
    ref=map_cache[(branch,participants[0])][0]; hdr=ref.header.copy();hdr.set_data_dtype(np.uint8)
    path=OUT/'masks'/f'{branch}_FisherZ_common_support_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask.astype(np.uint8),ref.affine,hdr),str(path));branch_masks[branch]=path

job_rows=[]; order_rows=[]; design_audit=[]
for model_index,(branch,design,role) in enumerate(specs):
    order=model_orders[(branch,design)]
    ddir=ATOG/'designs'/design
    for name in ('design.mat','design.con','design.fts','design_columns.tsv','contrast_key.tsv'):
        if not (ddir/name).exists():raise RuntimeError(f'Missing {name} for {design}')
    target_design=OUT/'designs'/design;target_design.mkdir(parents=True,exist_ok=True)
    for name in ('design.mat','design.con','design.fts','design_columns.tsv','contrast_key.tsv'):
        if not (target_design/name).exists():shutil.copy2(ddir/name,target_design/name)
    head,mat=parse_vest(ddir/'design.mat'); names=read_design_columns(ddir/'design_columns.tsv',mat.shape[1])
    if len(order)!=mat.shape[0]:raise RuntimeError(f'Order/design row count mismatch: {branch}/{design}')
    for i,p in enumerate(order):
        diff=float(np.max(np.abs(expected_vector(p,names)-mat[i,:])))
        design_audit.append({'branch':branch,'design_name':design,'row_index_1based':i+1,'participant_id':p,'maximum_absolute_design_difference':diff,'pass':int(diff<=1e-4)})
        if diff>1e-4:raise RuntimeError(f'Design row mismatch: {branch}/{design}/row{i+1}/{p}')
    imgs=[map_cache[(branch,p)][0] for p in order]; arrays=[map_cache[(branch,p)][1] for p in order]
    data4d=np.stack(arrays,axis=3); ref=imgs[0];hdr=ref.header.copy();hdr.set_data_shape(data4d.shape);hdr.set_data_dtype(np.float32)
    sdir=OUT/'stacks'/branch/design;sdir.mkdir(parents=True,exist_ok=True)
    stack_path=sdir/f'{branch}_{design}_FisherZ_4D.nii.gz';nib.save(nib.Nifti1Image(data4d,ref.affine,hdr),str(stack_path))
    sidecar=sdir/f'{branch}_{design}_FisherZ_4D_participant_order.tsv'
    side=[]
    for i,p in enumerate(order):
        source=selected[(branch,p)]
        side.append({'volume_index_1based':i+1,'volume_index_0based':i,'design_row_index_1based':i+1,'participant_id':p,'fisherz_source_map':str(source),'source_sha256':sha(source),'stack_path':str(stack_path),'branch':branch,'design_name':design})
        order_rows.append(side[-1])
    write_tsv(sidecar,side)
    check=nib.load(str(stack_path))
    for i,p in enumerate(order):
        if float(np.max(np.abs(np.asarray(check.dataobj[...,i],dtype=np.float32)-map_cache[(branch,p)][1])))>1e-6:raise RuntimeError(f'Stack round-trip mismatch: {branch}/{design}/{p}')
    outdir=OUT/'randomise'/branch/design;outdir.mkdir(parents=True,exist_ok=True)
    prefix=outdir/f'{branch}_{design}'
    job_rows.append({'array_index':model_index,'model_id':f'{branch}__{design}','modality':branch,'design_name':design,'analysis_roles':role+';branch_sensitivity','stack_path':str(stack_path),'participant_order_path':str(sidecar),'mask_path':str(branch_masks[branch]),'design_mat':str(target_design/'design.mat'),'design_con':str(target_design/'design.con'),'design_fts':str(target_design/'design.fts'),'contrast_key':str(target_design/'contrast_key.tsv'),'output_prefix':str(prefix),'randomise_directory':str(outdir),'n_participants':len(order),'n_permutations':5000})
write_tsv(OUT/'manifests'/'LNM_randomise_job_manifest.tsv',job_rows)
write_tsv(OUT/'manifests'/'all_LNM_stack_participant_orders.tsv',order_rows)
write_tsv(OUT/'audit'/'LNM_design_row_audit.tsv',design_audit)

shutil.copy2(V3/'tables'/'randomise_models_certified_existing_outputs.tsv',OUT/'manifests'/'certified_BCB_randomise_models.tsv')

quarantine=[]
for branch in ('LNM_BranchA','LNM_BranchB'):
    base=ATOG/'randomise'/branch
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir():quarantine.append({'path':str(d),'category':'legacy_invalid_LNM_randomise','status':'WITHDRAW_DO_NOT_INTERPRET','reason':'4D stack numerically matched lesion-weight/PV seed images rather than Fisher-Z connectivity maps'})
for pattern in ('*CLUSTER*SYMPTOM*','*cluster*symptom*','FINAL_RANDOMISE_CLUSTER_ANATOMY_SYMPTOM_PDF_PACK_*'):
    for p in RUN.glob(pattern):quarantine.append({'path':str(p),'category':'same_sample_cluster_symptom','status':'DESCRIPTIVE_ONLY_NOT_INFERENTIAL','reason':'Cluster selected and symptom association tested in the same participants; nested/external validation required'})
write_tsv(OUT/'legacy_quarantine'/'WITHDRAWN_AND_NONINFERENTIAL_OUTPUTS.tsv',quarantine,fields=['path','category','status','reason'])

qc=[
 {'check':'LNM_models_prepared','observed':len(job_rows),'expected':10 if ADD_TOP10 else 8,'pass':int(len(job_rows)==(10 if ADD_TOP10 else 8))},
 {'check':'all_design_rows_match','observed':sum(r['pass'] for r in design_audit),'expected':len(design_audit),'pass':int(all(r['pass']==1 for r in design_audit))},
 {'check':'all_FisherZ_maps_resolved','observed':len(selected),'expected':2*len(all_participants),'pass':int(len(selected)==2*len(all_participants))},
 {'check':'all_order_sidecars_written','observed':sum(Path(r['participant_order_path']).exists() for r in job_rows),'expected':len(job_rows),'pass':int(all(Path(r['participant_order_path']).exists() for r in job_rows))},
 {'check':'certified_BCB_manifest_present','observed':int((OUT/'manifests'/'certified_BCB_randomise_models.tsv').exists()),'expected':1,'pass':int((OUT/'manifests'/'certified_BCB_randomise_models.tsv').exists())},
]
write_tsv(OUT/'audit'/'FINAL_PREPARATION_QC.tsv',qc)
if not all(r['pass']==1 for r in qc):raise RuntimeError('Preparation QC failed')

with (OUT/'report'/'PREPARATION_REPORT.md').open('w') as h:
    h.write('# Bulletproof randomise preparation\n\n')
    h.write(f'- New Fisher-Z LNM models prepared: **{len(job_rows)}**\n- Certified BCB models retained: **{len(read_table(OUT/"manifests"/"certified_BCB_randomise_models.tsv"))}**\n- Legacy invalid LNM outputs quarantined: **{sum(r["category"]=="legacy_invalid_LNM_randomise" for r in quarantine)}**\n\n')
    h.write('Every new 4D LNM volume was matched to an explicit Fisher-Z source path, SHA256 hash, participant ID and design row. Original invalid LNM outputs were not modified.\n')
print('Preparation complete:',OUT)
PY

rm -f "$OUT/PREPARE_FAILED.flag"
cat > "$OUT/PREPARE_SUCCESS.flag" <<EOF
SUCCESS
completed=$(date --iso-8601=seconds 2>/dev/null || date)
job_manifest=$OUT/manifests/LNM_randomise_job_manifest.tsv
EOF
find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/audit/PREPARE_SHA256SUMS.txt"
echo "Preparation complete: $OUT"
SOURCE_PREPARE_PARTB_RANDOMISE_BULLETPROOF_V1_1_SH

cat > "$WORK/submit_partB_randomise_bulletproof_v1_1.sh" <<'SOURCE_SUBMIT_PARTB_RANDOMISE_BULLETPROOF_V1_1_SH'
#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "ERROR: do not source" >&2; return 2; fi
set -Eeuo pipefail
PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
ADD_LNM_TOP10="${ADD_LNM_TOP10:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${ROOT:-${RUN}/BULLETPROOF_RANDOMISE_${STAMP}}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-${PROJECT}/prepare_partB_randomise_bulletproof_v1_1.sh}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT}/run_one_partB_lnm_randomise_v1.sh}"
FINALISE_SCRIPT="${FINALISE_SCRIPT:-${PROJECT}/finalise_partB_randomise_bulletproof_v1.sh}"
for s in "$PREPARE_SCRIPT" "$RUN_SCRIPT" "$FINALISE_SCRIPT"; do [[ -f "$s" ]] || { echo "ERROR: missing $s" >&2; exit 1; }; done
mkdir -p "$PROJECT/slurm_logs" "$ROOT/logs/slurm"
NMODELS=8; [[ "$ADD_LNM_TOP10" == "1" ]] && NMODELS=10
PREP_JOB=$(sbatch --parsable --job-name=RandPrep --partition=nodes --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --time=08:00:00 --output="$ROOT/logs/slurm/prepare_%j.out" --error="$ROOT/logs/slurm/prepare_%j.err" --export=ALL,PROJECT="$PROJECT",RUN="$RUN",OUT="$ROOT",ADD_LNM_TOP10="$ADD_LNM_TOP10" --wrap="bash '$PREPARE_SCRIPT'")
ARRAY_JOB=$(sbatch --parsable --dependency=afterok:$PREP_JOB --kill-on-invalid-dep=yes --job-name=LNMrand --partition=nodes --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=24G --time=48:00:00 --array=0-$((NMODELS-1)) --output="$ROOT/logs/slurm/randomise_%A_%a.out" --error="$ROOT/logs/slurm/randomise_%A_%a.err" --export=ALL,ROOT="$ROOT" --wrap="bash '$RUN_SCRIPT'")
FINAL_JOB=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB --kill-on-invalid-dep=yes --job-name=RandFinal --partition=nodes --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --time=12:00:00 --output="$ROOT/logs/slurm/finalise_%j.out" --error="$ROOT/logs/slurm/finalise_%j.err" --export=ALL,PROJECT="$PROJECT",RUN="$RUN",ROOT="$ROOT" --wrap="bash '$FINALISE_SCRIPT'")
cat <<EOF
ROOT=$ROOT
Preparation job: $PREP_JOB
LNM randomise array: $ARRAY_JOB (0-$((NMODELS-1)))
Finalisation job: $FINAL_JOB

Monitor:
  squeue -j $PREP_JOB,$ARRAY_JOB,$FINAL_JOB
  tail -f $ROOT/logs/slurm/prepare_${PREP_JOB}.out
  tail -f $ROOT/logs/slurm/finalise_${FINAL_JOB}.out
EOF
SOURCE_SUBMIT_PARTB_RANDOMISE_BULLETPROOF_V1_1_SH

cat > "$WORK/run_one_partB_lnm_randomise_v1.sh" <<'SOURCE_RUN_ONE_PARTB_LNM_RANDOMISE_V1_SH'
#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "ERROR: do not source" >&2; return 2; fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT="${ROOT:?ROOT must point to the prepared bulletproof randomise directory}"
INDEX="${SLURM_ARRAY_TASK_ID:-${INDEX:-}}"
[[ -n "$INDEX" ]] || { echo "ERROR: array index unavailable" >&2; exit 1; }
MANIFEST="$ROOT/manifests/LNM_randomise_job_manifest.tsv"
[[ -f "$ROOT/PREPARE_SUCCESS.flag" && -f "$MANIFEST" ]] || { echo "ERROR: preparation incomplete" >&2; exit 1; }

module purge >/dev/null 2>&1 || true
module load fsl >/dev/null 2>&1 || true
command -v randomise >/dev/null 2>&1 || { echo "ERROR: FSL randomise unavailable" >&2; exit 1; }

mapfile -t FIELDS < <(python3 - "$MANIFEST" "$INDEX" <<'PY'
import csv,sys
path,index=sys.argv[1],int(sys.argv[2])
with open(path,newline='') as h:rows=list(csv.DictReader(h,delimiter='\t'))
if index<0 or index>=len(rows):raise SystemExit(f'Index {index} outside 0..{len(rows)-1}')
r=rows[index]
for k in ('model_id','stack_path','mask_path','design_mat','design_con','design_fts','output_prefix','randomise_directory','n_permutations'):
 print(r[k])
PY
)
MODEL_ID="${FIELDS[0]}"; STACK="${FIELDS[1]}"; MASK="${FIELDS[2]}"; DMAT="${FIELDS[3]}"; DCON="${FIELDS[4]}"; DFTS="${FIELDS[5]}"; PREFIX="${FIELDS[6]}"; RDIR="${FIELDS[7]}"; NPERM="${FIELDS[8]}"
mkdir -p "$RDIR" "$ROOT/logs/randomise"
LOG="$ROOT/logs/randomise/${MODEL_ID}.log"
exec > >(tee "$LOG") 2>&1

on_error(){ rc=$?; trap - ERR; echo -e "FAILED\nmodel=$MODEL_ID\nexit_code=$rc\ncommand=${BASH_COMMAND:-unknown}\nlog=$LOG" > "$RDIR/FAILED.flag"; exit "$rc"; }
trap on_error ERR

for p in "$STACK" "$MASK" "$DMAT" "$DCON" "$DFTS"; do [[ -f "$p" ]] || { echo "ERROR: missing $p" >&2; exit 1; }; done

SEED_ARGS=()
if randomise --help 2>&1 | grep -q -- '--seed'; then SEED_ARGS=(--seed=20260712); fi
CMD=(randomise -i "$STACK" -o "$PREFIX" -m "$MASK" -d "$DMAT" -t "$DCON" -f "$DFTS" -n "$NPERM" -T "${SEED_ARGS[@]}")
printf '%q ' "${CMD[@]}" > "$RDIR/randomise_command.sh"; echo >> "$RDIR/randomise_command.sh"
randomise --version > "$RDIR/randomise_version.txt" 2>&1 || true
"${CMD[@]}"

for i in {1..8}; do [[ -f "${PREFIX}_tfce_corrp_tstat${i}.nii.gz" ]] || { echo "ERROR: missing tstat${i} corrp" >&2; exit 1; }; done
[[ -f "${PREFIX}_tfce_corrp_fstat1.nii.gz" ]] || { echo "ERROR: missing fstat1 corrp" >&2; exit 1; }
sha256sum "$STACK" "$MASK" "$DMAT" "$DCON" "$DFTS" > "$RDIR/input_sha256.txt"
find "$RDIR" -maxdepth 1 -type f -name '*.nii.gz' -print0 | sort -z | xargs -0 sha256sum > "$RDIR/output_nifti_sha256.txt"
rm -f "$RDIR/FAILED.flag"
echo -e "SUCCESS\nmodel=$MODEL_ID\ncompleted=$(date --iso-8601=seconds 2>/dev/null || date)\npermutations=$NPERM" > "$RDIR/SUCCESS.flag"
echo "Completed $MODEL_ID"
SOURCE_RUN_ONE_PARTB_LNM_RANDOMISE_V1_SH

cat > "$WORK/finalise_partB_randomise_bulletproof_v1_1.sh" <<'SOURCE_FINALISE_PARTB_RANDOMISE_BULLETPROOF_V1_1_SH'
#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "ERROR: do not source" >&2; return 2; fi
set -Eeuo pipefail
umask 0027
export LC_ALL=C
export LANG=C

ROOT="${ROOT:?ROOT must point to bulletproof randomise directory}"
PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-${PROJECT}/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
PARTB="${PARTB:-${RUN}/MASTER_REBUILT_PART_B_20260710_215017}"
ATOG="${ATOG:-${RUN}/AtoG_spatial_followups_20260707_214137}"
DATASET="${DATASET:-${PARTB}/tables/model_dataset_REBUILT_structured_HCPex_LNM.tsv}"
CORRP_THRESHOLD="${CORRP_THRESHOLD:-0.95}"
NEAR_THRESHOLD="${NEAR_THRESHOLD:-0.90}"

mkdir -p "$ROOT"/{audit,clean_clusters,final_report,tables}
LOG="$ROOT/logs/finalise.log";exec > >(tee "$LOG") 2>&1
on_error(){ rc=$?; trap - ERR; echo -e "FAILED\nstage=finalise\nexit_code=$rc\ncommand=${BASH_COMMAND:-unknown}\nlog=$LOG" > "$ROOT/FINALISE_FAILED.flag"; exit "$rc"; };trap on_error ERR

module purge >/dev/null 2>&1 || true
module load apptainer >/dev/null 2>&1 || true
APPTAINER_BIN="${APPTAINER_BIN:-/opt/apps/pkg/tools/apptainer/1.3.6/bin/apptainer}"; [[ -x "$APPTAINER_BIN" ]] || APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
FMRIPREP_SIF="${FMRIPREP_SIF:-/opt/apps/pkg/applications/containers/fmriprep/25.1.3/fmriprep_25.1.3.sif}"
[[ -f "$ROOT/PREPARE_SUCCESS.flag" ]] || { echo "ERROR: preparation incomplete" >&2; exit 1; }

export ROOT RUN PARTB ATOG DATASET CORRP_THRESHOLD NEAR_THRESHOLD
"$APPTAINER_BIN" exec -B /mnt/scratch:/mnt/scratch -B /opt/apps:/opt/apps "$FMRIPREP_SIF" python - <<'PY'
import csv,hashlib,math,os,re
from collections import defaultdict
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT=Path(os.environ['ROOT']);RUN=Path(os.environ['RUN']);PARTB=Path(os.environ['PARTB']);ATOG=Path(os.environ['ATOG']);DATASET=Path(os.environ['DATASET'])
TH=float(os.environ['CORRP_THRESHOLD']);NEAR=float(os.environ['NEAR_THRESHOLD'])
LABELS={1:'CPSP_gt_SNP',2:'SNP_gt_CPSP',3:'CPSP_gt_NN',4:'NN_gt_CPSP',5:'CPSP_gt_all_nonCPSP_weighted',6:'all_nonCPSP_weighted_gt_CPSP',7:'NN_gt_SNP',8:'SNP_gt_NN'}

def read(path):
 lines=path.read_text(errors='replace').splitlines();sep='\t' if lines and '\t' in lines[0] else ','
 with path.open(newline='') as h:return list(csv.DictReader(h,delimiter=sep))
def write(path,rows,fields=None):
 path.parent.mkdir(parents=True,exist_ok=True);fields=fields or (list(rows[0]) if rows else [])
 with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows)
def norm(x):return re.sub(r'[^a-z0-9]','',str(x).lower())
def col(rows,aliases):
 d={norm(k):k for k in rows[0]} if rows else {}
 for a in aliases:
  if norm(a) in d:return d[norm(a)]
 return None
def vest(path):
 lines=path.read_text(errors='replace').splitlines();head={};start=None
 for i,l in enumerate(lines):
  s=l.strip()
  if s.startswith('/') and ' ' in s:k,v=s.split(None,1);head[k]=v
  if s=='/Matrix':start=i+1;break
 if start is None:raise RuntimeError(f'No matrix: {path}')
 return head,np.asarray([[float(x) for x in l.split()] for l in lines[start:] if l.strip()],float)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def holm(ps):
 n=len(ps);order=np.argsort(ps);out=np.ones(n);running=0
 for rank,idx in enumerate(order):
  val=(n-rank)*ps[idx];running=max(running,val);out[idx]=min(1,running)
 return out
def bh(ps):
 n=len(ps);order=np.argsort(ps)[::-1];out=np.ones(n);running=1
 for revrank,idx in enumerate(order):
  rank=n-revrank;val=ps[idx]*n/rank;running=min(running,val);out[idx]=min(1,running)
 return out

lnm=read(ROOT/'manifests'/'LNM_randomise_job_manifest.tsv')
for r in lnm:
 if not Path(r['randomise_directory']).joinpath('SUCCESS.flag').exists():raise RuntimeError(f'Incomplete LNM model: {r["model_id"]}')

bcb=read(ROOT/'manifests'/'certified_BCB_randomise_models.tsv')
if len(bcb)!=10:raise RuntimeError(f'Expected 10 certified BCB models, found {len(bcb)}')
repair_manifest_path=ROOT/'manifests'/'repaired_BCB_randomise_manifest.tsv'
repairs={}
if repair_manifest_path.exists():
 repair_rows=read(repair_manifest_path)
 repairs={(r['modality'],r['design_name']):r for r in repair_rows}
 for key,r in repairs.items():
  if not (Path(r['randomise_directory'])/'SUCCESS.flag').exists():raise RuntimeError(f'Repaired BCB model incomplete: {key[0]}/{key[1]}')

contrast=[];designs=sorted({r['design_name'] for r in lnm+bcb})
for d in designs:
 ddir=(ROOT/'designs'/d) if (ROOT/'designs'/d).exists() else (ATOG/'designs'/d)
 ch,cm=vest(ddir/'design.con');fh,fm=vest(ddir/'design.fts');key=read(ddir/'contrast_key.tsv')
 ki=col(key,('contrast_index','index'));kn=col(key,('contrast_name','name','contrast'));labels={int(float(r[ki])):r[kn] for r in key}
 if cm.shape[0]!=8 or fm.shape!=(1,8):raise RuntimeError(f'Unexpected contrast dimensions: {d}')
 pairs=((1,2),(3,4),(5,6),(7,8))
 for i in range(1,9):
  rev=any(i in p and np.allclose(cm[p[0]-1],-cm[p[1]-1],atol=1e-8) for p in pairs)
  lab=labels.get(i)==LABELS[i]
  contrast.append({'design_name':d,'stat_type':'tstat','stat_index':i,'contrast_label':labels.get(i,''),'expected_label':LABELS[i],'contrast_vector':';'.join(f'{x:.12g}' for x in cm[i-1]),'label_pass':int(lab),'reverse_pair_pass':int(rev),'pass':int(lab and rev)})
 contrast.append({'design_name':d,'stat_type':'fstat','stat_index':1,'contrast_label':'omnibus_group_F_test','expected_label':'omnibus_group_F_test','contrast_vector':';'.join(f'{x:.12g}' for x in fm[0]),'label_pass':1,'reverse_pair_pass':'','pass':int(np.linalg.matrix_rank(cm[np.where(np.abs(fm[0])>.5)[0],:])==2)})
write(ROOT/'tables'/'contrast_name_direction_audit.tsv',contrast)
if not all(int(r['pass'])==1 for r in contrast):raise RuntimeError('Contrast audit failed')

models=[]
for r in lnm:
 models.append({'modality':r['modality'],'design_name':r['design_name'],'analysis_roles':r['analysis_roles'],'randomise_directory':r['randomise_directory'],'output_prefix':r['output_prefix'],'n_participants':r['n_participants'],'source':'new_clean_FisherZ_LNM','participant_order_path':r['participant_order_path']})
for r in bcb:
 m=r['modality'];d=r['design_name'];roles=[]
 if m=='BCB_BranchB' and d=='stroke_all_minimal':roles.append('full_sample_primary_spatial_followup')
 if 'BranchA' in m or 'BranchC' in m:roles.append('branch_sensitivity')
 if 'top10' in d:roles.append('lesion_size_sensitivity')
 if 'age_sex' in d:roles.append('age_sex_sensitivity')
 if 'strict' in d:roles.append('strict_control_sensitivity')
 if not roles:roles.append('other_spatial_followup')
 key=(m,d)
 if key in repairs:
  repair=repairs[key];rdir=Path(repair['randomise_directory']);prefix=repair['output_prefix'];source='clean_repaired_certified_BCB'
 else:
  rdir=Path(r['randomise_directory']);prefix_path=next(iter(rdir.glob('*_tfce_corrp_tstat1.nii.gz')),None)
  if not prefix_path:raise RuntimeError(f'No BCB corrp map: {m}/{d}')
  prefix=str(prefix_path).replace('_tfce_corrp_tstat1.nii.gz','');source='certified_existing_BCB'
 models.append({'modality':m,'design_name':d,'analysis_roles':';'.join(roles),'randomise_directory':str(rdir),'output_prefix':str(prefix),'n_participants':'','source':source,'participant_order_path':r.get('participant_order_sidecar','')})

master=[];clusters=[];structure=ndimage.generate_binary_structure(3,3)
for model in models:
 rows=[]
 for st,i,label in [('tstat',i,LABELS[i]) for i in range(1,9)]+[('fstat',1,'omnibus_group_F_test')]:
  path=Path(f"{model['output_prefix']}_tfce_corrp_{st}{i}.nii.gz")
  if not path.exists():raise RuntimeError(f'Missing corrp map: {path}')
  img=nib.load(str(path));data=np.asarray(img.dataobj,dtype=np.float32);finite=data[np.isfinite(data)];mx=float(np.max(finite));p=1-mx;mask=np.isfinite(data)&(data>TH);nvox=int(mask.sum());ncl=0;status='NULL_TFCE_FWE'
  if nvox:
   status='TFCE_FWE_SIGNIFICANT_CORRP_GT_0P95';lab,ncl=ndimage.label(mask,structure=structure);cdir=ROOT/'clean_clusters'/model['modality']/model['design_name']/f'{st}{i}_{label}';cdir.mkdir(parents=True,exist_ok=True)
   for cid in range(1,ncl+1):
    cm=lab==cid;nv=int(cm.sum());peak=np.unravel_index(int(np.argmax(np.where(cm,data,-np.inf))),data.shape);xyz=nib.affines.apply_affine(img.affine,np.asarray(peak));vv=float(abs(np.linalg.det(img.affine[:3,:3])));cp=cdir/f'cluster_{cid:03d}_corrp_gt_0p95.nii.gz';hdr=img.header.copy();hdr.set_data_dtype(np.uint8);nib.save(nib.Nifti1Image(cm.astype(np.uint8),img.affine,hdr),str(cp))
    clusters.append({'modality':model['modality'],'design_name':model['design_name'],'analysis_roles':model['analysis_roles'],'stat_type':st,'stat_index':i,'contrast_label':label,'cluster_id':cid,'n_voxels':nv,'cluster_volume_mm3':nv*vv,'max_corrp':float(data[cm].max()),'corrected_p_equivalent':1-float(data[cm].max()),'peak_x_mm':float(xyz[0]),'peak_y_mm':float(xyz[1]),'peak_z_mm':float(xyz[2]),'cluster_mask_path':str(cp),'same_sample_cluster_symptom_inference_permitted':0})
  elif mx>NEAR:status='NEAR_THRESHOLD_NOT_SIGNIFICANT'
  row={**model,'stat_type':st,'stat_index':i,'contrast_label':label,'corrp_map_path':str(path),'corrp_sha256':sha(path),'max_corrp':mx,'minimum_corrected_p':p,'n_voxels_corrp_gt_0p95':nvox,'n_corrected_clusters':ncl,'result_status':status,'interpret_as_significant':int(nvox>0),'corrp_rule':'corrp > .95 = TFCE FWE corrected p < .05','near_threshold_rule':'.90 < corrp <= .95 is not significant','same_sample_cluster_symptom_rule':'No inferential regression without nested or external validation'}
  rows.append(row)
 ps=np.asarray([r['minimum_corrected_p'] for r in rows]);hp=holm(ps);bq=bh(ps)
 for j,r in enumerate(rows):r['holm_p_across_9_contrasts']=float(hp[j]);r['BH_q_across_9_contrasts']=float(bq[j]);master.append(r)
write(ROOT/'tables'/'MASTER_RANDOMISE_ALL_MODELS_ALL_9_CONTRASTS.tsv',master)
write(ROOT/'tables'/'CLEAN_CORRECTED_CLUSTERS.tsv',clusters,fields=['modality','design_name','analysis_roles','stat_type','stat_index','contrast_label','cluster_id','n_voxels','cluster_volume_mm3','max_corrp','corrected_p_equivalent','peak_x_mm','peak_y_mm','peak_z_mm','cluster_mask_path','same_sample_cluster_symptom_inference_permitted'])
write(ROOT/'tables'/'NEAR_THRESHOLD_NOT_SIGNIFICANT.tsv',[r for r in master if r['result_status']=='NEAR_THRESHOLD_NOT_SIGNIFICANT'],fields=list(master[0]))
write(ROOT/'tables'/'SIGNIFICANT_CONTRASTS.tsv',[r for r in master if r['interpret_as_significant']==1],fields=list(master[0]))

manifest=read(ROOT/'manifests'/'FisherZ_participant_map_manifest.tsv');by=defaultdict(dict)
for r in manifest:by[r['participant_id']][r['branch']]=Path(r['fisherz_map_path'])
maskA=np.asarray(nib.load(str(ROOT/'masks'/'LNM_BranchA_FisherZ_common_support_mask.nii.gz')).dataobj)>0
maskB=np.asarray(nib.load(str(ROOT/'masks'/'LNM_BranchB_FisherZ_common_support_mask.nii.gz')).dataobj)>0
common=maskA&maskB
rel=[]
for p,x in sorted(by.items()):
 if not {'LNM_BranchA','LNM_BranchB'}<=set(x):continue
 a=np.asarray(nib.load(str(x['LNM_BranchA'])).dataobj,dtype=float)[common];b=np.asarray(nib.load(str(x['LNM_BranchB'])).dataobj,dtype=float)[common]
 rel.append({'participant_id':p,'n_common_voxels':int(common.sum()),'spatial_pearson_r':float(np.corrcoef(a,b)[0,1]),'mean_absolute_difference':float(np.mean(np.abs(a-b))),'RMSE':float(np.sqrt(np.mean((a-b)**2))),'sign_agreement_fraction':float(np.mean(np.sign(a)==np.sign(b)))})
write(ROOT/'tables'/'LNM_BranchA_vs_B_FisherZ_spatial_reliability.tsv',rel)

lookup={(r['modality'],r['design_name'],r['stat_type'],int(r['stat_index'])):r for r in master}
conc=[]
for d in sorted({r['design_name'] for r in lnm}):
 for st,i,label in [('tstat',i,LABELS[i]) for i in range(1,9)]+[('fstat',1,'omnibus_group_F_test')]:
  ka=('LNM_BranchA',d,st,i);kb=('LNM_BranchB',d,st,i)
  if ka not in lookup or kb not in lookup:continue
  a=np.asarray(nib.load(lookup[ka]['corrp_map_path']).dataobj,dtype=float)[common];b=np.asarray(nib.load(lookup[kb]['corrp_map_path']).dataobj,dtype=float)[common]
  conc.append({'design_name':d,'stat_type':st,'stat_index':i,'contrast_label':label,'corrp_map_spatial_pearson_r':float(np.corrcoef(a,b)[0,1]),'mean_absolute_corrp_difference':float(np.mean(np.abs(a-b))),'BranchA_max_corrp':lookup[ka]['max_corrp'],'BranchB_max_corrp':lookup[kb]['max_corrp']})
write(ROOT/'tables'/'LNM_BranchA_vs_B_randomise_result_concordance.tsv',conc)

reliability=[]
for f in PARTB.rglob('*'):
 if f.is_file() and any(t in f.name.lower() for t in ('hc19','normative','control_loo','leave_one_control')) and any(t in f.name.lower() for t in ('reliab','loo','stability','summary','table')):
  reliability.append({'path':str(f),'sha256':sha(f),'status':'existing_normative_reliability_evidence_archived_by_reference'})
write(ROOT/'tables'/'EXISTING_HC19_NORMATIVE_RELIABILITY_FILES.tsv',reliability,fields=['path','sha256','status'])

registry=[]
for m in models:
 rs=[r for r in master if r['modality']==m['modality'] and r['design_name']==m['design_name']]
 registry.append({**m,'n_contrasts_reported':len(rs),'n_significant_contrasts':sum(r['interpret_as_significant'] for r in rs),'n_near_threshold_not_significant':sum(r['result_status']=='NEAR_THRESHOLD_NOT_SIGNIFICANT' for r in rs),'all_nine_reported':int(len(rs)==9)})
write(ROOT/'tables'/'RANDOMISE_MODEL_REGISTRY.tsv',registry)
qc=[
 {'check':'new_LNM_models_complete','observed':len(lnm),'expected':len(lnm),'pass':1},
 {'check':'certified_BCB_models_included','observed':len(bcb),'expected':10,'pass':int(len(bcb)==10)},
 {'check':'all_models_have_9_contrasts','observed':sum(r['all_nine_reported'] for r in registry),'expected':len(registry),'pass':int(all(r['all_nine_reported']==1 for r in registry))},
 {'check':'near_threshold_never_significant','observed':sum(r['interpret_as_significant'] for r in master if r['result_status']=='NEAR_THRESHOLD_NOT_SIGNIFICANT'),'expected':0,'pass':int(all(r['interpret_as_significant']==0 for r in master if r['result_status']=='NEAR_THRESHOLD_NOT_SIGNIFICANT'))},
 {'check':'same_sample_cluster_symptom_inference_generated','observed':0,'expected':0,'pass':1},
 {'check':'contrast_direction_audit','observed':sum(int(r['pass']) for r in contrast),'expected':len(contrast),'pass':int(all(int(r['pass'])==1 for r in contrast))},
 {'check':'minimum_permutation_p_resolution','observed':1/5001,'expected':'0.00019996','pass':1},
]
write(ROOT/'audit'/'FINAL_BULLETPROOF_RANDOMISE_QC.tsv',qc)
if not all(r['pass']==1 for r in qc):raise RuntimeError('Final QC failed')

sig=[r for r in master if r['interpret_as_significant']==1];near=[r for r in master if r['result_status']=='NEAR_THRESHOLD_NOT_SIGNIFICANT']
with (ROOT/'final_report'/'PART_B_RANDOMISE_BULLETPROOF_REPORT.md').open('w') as h:
 h.write('# Part B randomise final report\n\n')
 h.write(f'- Clean Fisher-Z LNM models: **{len(lnm)}**\n- Certified BCB models: **{len(bcb)}**\n- Total models: **{len(models)}**\n- Total contrast rows: **{len(master)}**\n- TFCE-FWE significant contrast maps: **{len(sig)}**\n- Near-threshold maps retained as non-significant: **{len(near)}**\n\n')
 h.write('## Inference rules\n\n`corrp > .95` was interpreted as TFCE family-wise-error corrected `p < .05`. All eight directional t-contrasts and the omnibus F-test were reported for every model, including null results. Values between `.90` and `.95` were not interpreted as significant. Within-model Holm and Benjamini-Hochberg adjustments across the nine contrast-level minimum corrected p-values are supplied as supplementary multiplicity checks.\n\n')
 h.write('## Provenance\n\nAll new LNM stacks were built from explicit participant Fisher-Z maps with SHA256 hashes and participant-order sidecars. The ten BCB models were included only from the successful stack/design certification audit. Legacy lesion-seed LNM randomise outputs are withdrawn.\n\n')
 h.write('## Reliability and sensitivity\n\nBranch A versus Branch B Fisher-Z spatial agreement is reported per participant, and concordance of Branch A versus Branch B corrected-probability maps is reported for matching models and contrasts. Sensitivity families include age/sex adjustment, strict-control exclusion, low-coverage exclusion and lesion-size exclusion, alongside Branch A/Branch B and BCB Branch C comparisons. Existing HC19 normative-control reliability files are indexed separately.\n\n')
 h.write('## Cluster guardrail\n\nClusters were extracted only for anatomical description and figures. No inferential cluster-to-symptom regression was generated because the clusters were selected in the same participants. Such inference requires nested cross-validation or an independent validation sample.\n\n')
 h.write('## Significant contrasts\n\n')
 if not sig:h.write('No contrast contained voxels with `corrp > .95`.\n')
 else:
  h.write('| Modality | Design | Contrast | Max corrp | Min corrected p | Voxels | Holm p across 9 |\n|---|---|---|---:|---:|---:|---:|\n')
  for r in sorted(sig,key=lambda x:(x['modality'],x['design_name'],x['stat_type'],int(x['stat_index']))):h.write(f"| {r['modality']} | {r['design_name']} | {r['contrast_label']} | {float(r['max_corrp']):.6f} | {float(r['minimum_corrected_p']):.6f} | {r['n_voxels_corrp_gt_0p95']} | {float(r['holm_p_across_9_contrasts']):.6f} |\n")
print('Finalisation complete:',ROOT)
PY

rm -f "$ROOT/FINALISE_FAILED.flag"
echo -e "SUCCESS\ncompleted=$(date --iso-8601=seconds 2>/dev/null || date)\nreport=$ROOT/final_report/PART_B_RANDOMISE_BULLETPROOF_REPORT.md" > "$ROOT/SUCCESS.flag"
find "$ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/audit/FINAL_SHA256SUMS.txt"
echo "Final report: $ROOT/final_report/PART_B_RANDOMISE_BULLETPROOF_REPORT.md"
SOURCE_FINALISE_PARTB_RANDOMISE_BULLETPROOF_V1_1_SH

cat > "$WORK/cpsp_palm_pipeline_v1.py" <<'SOURCE_CPSP_PALM_PIPELINE_V1_PY'
#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,html,math,os,re,subprocess,sys,tempfile
from datetime import datetime
from pathlib import Path

TOKENS=(
 'strict_nonCPSP_exclude_NN_DN4pos_minimal',
 'stroke_all_age_sex_adjusted',
 'stroke_all_minimal',
)

def run(cmd,check=True,timeout=300):
 r=subprocess.run(cmd,text=True,capture_output=True,check=check,timeout=timeout)
 return r.stdout.strip()

def safe(cmd):
 try:return run(cmd)
 except Exception:return ''

def vest(path,key):
 if not path or not Path(path).exists(): return 0
 pat=re.compile(rf'^/{re.escape(key)}\s+(\d+)')
 for line in Path(path).read_text(errors='replace').splitlines():
  m=pat.match(line.strip())
  if m:return int(m.group(1))
 return 0

def dim4(path):
 try:return int(float(safe(['fslval',str(path),'dim4']).split()[0]))
 except Exception:return 0

def dims3(path):
 vals=[]
 for key in ('dim1','dim2','dim3'):
  try:vals.append(int(float(safe(['fslval',str(path),key]).split()[0])))
  except Exception:return ()
 return tuple(vals)

def contrast_names(path,n,prefix):
 names={}
 if path and Path(path).exists():
  pat=re.compile(r'^/ContrastName(\d+)\s+(.+)$')
  for line in Path(path).read_text(errors='replace').splitlines():
   m=pat.match(line.strip())
   if m:names[int(m.group(1))]=m.group(2).strip()
 return [names.get(i,f'{prefix}{i}') for i in range(1,n+1)]

def choose_randomise_root(run_root):
 hits=[]
 for flag in run_root.rglob('SUCCESS.flag'):
  low=str(flag).lower()
  if 'palm' in low or 'cpsp_comprehensive' in low:continue
  score=(40 if 'bulletproof' in low else 0)+(30 if 'randomise' in low else 0)+(20 if 'final' in low else 0)
  hits.append((score,flag.stat().st_mtime,flag.parent))
 if not hits:raise RuntimeError(f'No completed randomise finaliser SUCCESS.flag beneath {run_root}')
 hits.sort(reverse=True)
 return hits[0][2]

def nearby(directory,root):
 levels=[directory]
 p=directory.parent
 while p!=root.parent and len(levels)<4:
  levels.append(p)
  if p==root:break
  p=p.parent
 out=[]
 for level in levels:
  if not level.exists():continue
  for x in level.iterdir():
   if x.is_file():out.append(x)
   elif x.is_dir() and x.name.lower() in {'design','input','inputs','stats','randomise','final'}:
    out.extend(y for y in x.rglob('*') if y.is_file())
 return list(dict.fromkeys(out))

def input_score(path,n):
 if dim4(path)!=n:return -9999
 name=path.name.lower(); score=100
 for token,pts in [('input',25),('merged',20),('stack',20),('4d',20),('bcb',10),('final',8)]:
  if token in name:score+=pts
 for token in ('tstat','fstat','corrp','mask','mean','var','tfce','cluster','cope','zstat'):
  if token in name:score-=80
 return score

def mask_score(path,refdims):
 if dim4(path) not in (0,1):return -9999
 if refdims and dims3(path) and dims3(path)!=refdims:return -9999
 name=path.name.lower();score=0
 if 'mask' in name:score+=30
 if name in ('mask.nii','mask.nii.gz'):score+=30
 if 'final' in name:score+=5
 return score

def randomise_maps(directory):
 out=[]
 for level in [directory,directory.parent,*list(directory.parents)[:2]]:
  if not level.exists():continue
  out += list(level.rglob('*tfce_corrp_tstat*.nii*'))
  out += list(level.rglob('*tfce_corrp_fstat*.nii*'))
 return [p.resolve() for p in dict.fromkeys(out) if 'palm' not in str(p).lower()]

def command_text(directory):
 chunks=[]
 for level in (directory,directory.parent):
  if not level.exists():continue
  for p in level.iterdir():
   if p.is_file() and p.stat().st_size<5_000_000 and any(t in p.name.lower() for t in ('command','cmd','log','script')):
    txt=p.read_text(errors='replace')
    if 'randomise' in txt:chunks.append(txt)
 return '\n'.join(chunks)

def command_path(text,flags,directory):
 for flag in flags:
  m=re.search(rf'(?:^|\s){re.escape(flag)}\s+([^\s\\]+)',text)
  if m:
   p=Path(m.group(1).strip("'\"")); p=p if p.is_absolute() else directory/p
   if p.exists():return str(p.resolve())
 return ''

def select_model(root,token,audit):
 dirs=sorted({p.parent for p in root.rglob('*.mat') if 'design' in p.name.lower() and token.lower() in str(p).lower()})
 valid=[]
 for d in dirs:
  files=nearby(d,root)
  mats=[p for p in files if p.suffix=='.mat']; cons=[p for p in files if p.suffix=='.con']
  if not mats or not cons:continue
  mats.sort(key=lambda p:(p.name=='design.mat','design' in p.name.lower()),reverse=True)
  cons.sort(key=lambda p:(p.name=='design.con'),reverse=True)
  mat,con=mats[0],cons[0]; n=vest(mat,'NumPoints'); nt=vest(con,'NumContrasts')
  if not n or not nt:continue
  niftis=[p for p in files if str(p).lower().endswith(('.nii','.nii.gz'))]
  ranked=sorted(((input_score(p,n),p) for p in niftis),reverse=True)
  if not ranked or ranked[0][0]<0:continue
  inp=ranked[0][1]; ref=dims3(inp)
  masks=sorted(((mask_score(p,ref),p) for p in niftis if p!=inp),reverse=True)
  if not masks or masks[0][0]<10:continue
  mask=masks[0][1]
  fts=sorted([p for p in files if p.suffix=='.fts'],key=lambda p:p.name=='design.fts',reverse=True)
  fts=fts[0] if fts else None; nf=vest(fts,'NumContrasts') if fts else 0
  rmaps=randomise_maps(d); text=command_text(d)
  eb=command_path(text,('-e','-eb','--exchangeability_blocks'),d)
  vg=command_path(text,('-vg',),d)
  if not eb:
   grps=[p for p in files if p.suffix=='.grp']
   if grps:
    vals=set()
    for line in grps[0].read_text(errors='replace').splitlines():
     if not line.startswith('/') and line.strip():vals.update(line.split())
    if len(vals)<=1:eb=str(grps[0].resolve())
  rdir=Path(os.path.commonpath([str(p.parent) for p in rmaps])).resolve() if rmaps else d.resolve()
  row=dict(model_token=token,model_name=token,model_directory=str(d.resolve()),input_4d=str(inp.resolve()),mask=str(mask.resolve()),design_mat=str(mat.resolve()),design_con=str(con.resolve()),design_fts=str(fts.resolve()) if fts else '',exchangeability_blocks=eb,variance_groups=vg,randomise_directory=str(rdir),npoints=str(n),nwaves=str(vest(mat,'NumWaves')),n_t_contrasts=str(nt),n_f_contrasts=str(nf),randomise_corrp_maps=str(len(rmaps)),candidate_score=str(100+min(len(rmaps),20)+(10 if 'final' in str(d).lower() else 0)))
  valid.append(row);audit.append({'model_token':token,'candidate_directory':str(d),'input':str(inp),'design':str(mat),'mask':str(mask),'randomise_maps':str(len(rmaps))})
 if not valid:return None
 valid.sort(key=lambda r:int(r['candidate_score']),reverse=True)
 return valid[0]

def discover(args):
 run_root=args.run.resolve(); root=args.randomise_root.resolve() if args.randomise_root else choose_randomise_root(run_root)
 out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
 audit=[];rows=[]
 for token in args.model_token or TOKENS:
  row=select_model(root,token,audit)
  if row is None:raise RuntimeError(f'No complete final model found for {token} beneath {root}')
  rows.append(row)
 if len({(r['input_4d'],r['design_mat'],r['design_con']) for r in rows})!=len(rows):raise RuntimeError('Duplicate models discovered')
 fields=['model_index','model_token','model_name','model_directory','input_4d','mask','design_mat','design_con','design_fts','exchangeability_blocks','variance_groups','randomise_directory','npoints','nwaves','n_t_contrasts','n_f_contrasts','randomise_corrp_maps','candidate_score','randomise_final_root','discovered_at']
 now=datetime.now().astimezone().isoformat()
 for i,r in enumerate(rows):r.update(model_index=str(i),randomise_final_root=str(root),discovered_at=now)
 with out.open('w',newline='') as h:
  w=csv.DictWriter(h,fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
 ap=out.with_name(out.stem+'_candidate_audit.tsv')
 with ap.open('w',newline='') as h:
  f=['model_token','candidate_directory','input','design','mask','randomise_maps'];w=csv.DictWriter(h,f,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(audit)
 print(f'Randomise final root: {root}\nPALM models selected: {len(rows)}\nManifest: {out}\nCandidate audit: {ap}')
 return 0

def rng(path):
 p=safe(['fslstats',str(path),'-R']).split()
 try:return float(p[0]),float(p[1])
 except:return math.nan,math.nan

def threshold(path,thr,tmp):
 mask=tmp/(path.name.replace('.nii.gz','').replace('.nii','')+f'_thr{thr}.nii.gz')
 run(['fslmaths',str(path),'-thr',str(thr),'-bin',str(mask)])
 p=safe(['fslstats',str(mask),'-V']).split()
 return (int(float(p[0])),float(p[1]),mask) if len(p)>=2 else (0,0.0,mask)

def corr(a,b):
 nums=re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?',safe(['fslcc',str(a),str(b)]))
 try:return float(nums[-1])
 except:return math.nan

def dice(a,b,tmp):
 av=float(safe(['fslstats',str(a),'-V']).split()[0]);bv=float(safe(['fslstats',str(b),'-V']).split()[0])
 if av+bv==0:return 1.0
 inter=tmp/'inter.nii.gz';run(['fslmaths',str(a),'-mul',str(b),str(inter)])
 iv=float(safe(['fslstats',str(inter),'-V']).split()[0]);return 2*iv/(av+bv)

def parse_palm(path):
 m=re.search(r'_(vox|tfce)_(tstat|fstat)_(uncp|fwep|cfwep|fdrp)(?:_m\d+)?(?:_d\d+)?(?:_c(\d+))?\.nii(?:\.gz)?$',path.name,re.I)
 return None if not m else (m.group(1).lower(),m.group(2).lower(),m.group(3).lower(),int(m.group(4) or 1))

def palm_map(model_dir,unit,stat,family,index):
 hits=[]
 for p in model_dir.rglob('*.nii*'):
  q=parse_palm(p)
  if q==(unit,stat,family,index):hits.append(p)
 return max(hits,key=lambda p:p.stat().st_mtime) if hits else None

def rand_map(directory,stat,index):
 hits=[p for p in directory.rglob(f'*tfce_corrp_{stat}{index}.nii*') if 'palm' not in str(p).lower()]
 return max(hits,key=lambda p:(10 if 'final' in str(p).lower() else 0,p.stat().st_mtime)) if hits else None

def cluster_table(path,thr,out):
 r=subprocess.run(['cluster',f'--in={path}',f'--thresh={thr}','--mm'],text=True,capture_output=True)
 out.write_text(r.stdout+('\nSTDERR:\n'+r.stderr if r.stderr else ''))
 rows=[x for x in r.stdout.splitlines() if x.strip() and not x.lower().startswith('cluster')]
 return len(rows),rows[0] if rows else ''

def render(path,out,thr,standard):
 out.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory() as td:
  mask=Path(td)/'mask.nii.gz';subprocess.run(['fslmaths',str(path),'-thr',str(thr),'-bin',str(mask)],capture_output=True)
  under=standard if standard and standard.exists() else path
  r=subprocess.run(['slicer',str(under),str(mask),'-s','2','-S','2','1600',str(out)],capture_output=True)
  return r.returncode==0 and out.exists()

def html_table(rows,cols):
 s=['<div class="table-wrap"><table><thead><tr>']+[f'<th>{html.escape(c)}</th>' for c in cols]+['</tr></thead><tbody>']
 for r in rows:s+=['<tr>']+[f'<td>{html.escape(str(r.get(c,"")))}</td>' for c in cols]+['</tr>']
 return ''.join(s+['</tbody></table></div>'])

def report(args):
 root=args.palm_root.resolve();manifest=root/'audit/PALM_INPUT_MANIFEST.tsv'
 with manifest.open(newline='') as h:models=list(csv.DictReader(h,delimiter='\t'))
 report=root/'report';tabs=report/'tables';figs=report/'figures';clusters=report/'clusters'
 for d in (report,tabs,figs,clusters):d.mkdir(parents=True,exist_ok=True)
 std=None;fsldir=os.environ.get('FSLDIR','')
 if fsldir:
  p=Path(fsldir)/'data/standard/MNI152_T1_2mm_brain.nii.gz';std=p if p.exists() else None
 rows=[];qc=[];complete=True
 for m in models:
  name=m['model_name'];mdir=root/'models'/name;ok=(mdir/'SUCCESS.flag').exists();complete&=ok
  qc.append({'model':name,'success':str(ok),'failed':str((mdir/'FAILED.flag').exists()),'log':str(mdir/'logs/palm.log'),'command':str(mdir/'audit/PALM_COMMAND.sh'),'output_maps':str(len(list((mdir/'outputs').glob('*.nii*'))) if (mdir/'outputs').exists() else 0)})
  if not ok:continue
  nt,nf=int(m['n_t_contrasts']),int(m['n_f_contrasts']);tn=contrast_names(Path(m['design_con']),nt,'t');fn=contrast_names(Path(m['design_fts']),nf,'F') if m['design_fts'] else []
  rdir=Path(m['randomise_directory'])
  for stat,count,names in (('tstat',nt,tn),('fstat',nf,fn)):
   for idx in range(1,count+1):
    row={'model':name,'statistic':stat,'contrast_index':str(idx),'contrast_name':names[idx-1]}
    with tempfile.TemporaryDirectory() as td:
     tmp=Path(td);masks={}
     for fam in ('fwep','cfwep','fdrp','uncp'):
      p=palm_map(mdir,'tfce',stat,fam,idx);row[f'PALM_{fam}_map']=str(p) if p else ''
      if p:
       mx=rng(p)[1];v,mm,mask=threshold(p,args.threshold,tmp);masks[fam]=mask
       row[f'PALM_{fam}_max_1mp']=f'{mx:.6g}';row[f'PALM_{fam}_voxels']=str(v);row[f'PALM_{fam}_mm3']=f'{mm:.6g}'
      else:
       row[f'PALM_{fam}_max_1mp']=row[f'PALM_{fam}_voxels']=row[f'PALM_{fam}_mm3']=''
     rp=rand_map(rdir,stat,idx);row['randomise_corrp_map']=str(rp) if rp else ''
     if rp:
      rmx=rng(rp)[1];rv,rmm,rmask=threshold(rp,args.threshold,tmp);row.update(randomise_max_1mp=f'{rmx:.6g}',randomise_voxels=str(rv),randomise_mm3=f'{rmm:.6g}')
      pf=palm_map(mdir,'tfce',stat,'fwep',idx)
      if pf and 'fwep' in masks:
       row['randomise_PALM_fwep_dice']=f'{dice(rmask,masks["fwep"],tmp):.4f}';row['randomise_PALM_fwep_map_correlation']=f'{corr(rp,pf):.4f}'
     else:row.update(randomise_max_1mp='',randomise_voxels='',randomise_mm3='',randomise_PALM_fwep_dice='',randomise_PALM_fwep_map_correlation='')
     cp=palm_map(mdir,'tfce',stat,'cfwep',idx);cc=0;peak='';shot=''
     if cp:
      cc,peak=cluster_table(cp,args.threshold,clusters/f'{name}_{stat}_c{idx}_cfwep_clusters.txt')
      v=int(row.get('PALM_cfwep_voxels','0') or 0);mx=float(row.get('PALM_cfwep_max_1mp','0') or 0)
      if v>0 or mx>=args.near_threshold:
       sp=figs/f'{name}_{stat}_c{idx}_PALM_cfwep.png';render(cp,sp,args.threshold if v>0 else args.near_threshold,std);shot=str(sp) if sp.exists() else ''
     row['PALM_cfwep_clusters']=str(cc);row['PALM_cfwep_peak_row']=peak;row['screenshot']=shot
     rs=int(row.get('randomise_voxels','0') or 0)>0;pw=int(row.get('PALM_fwep_voxels','0') or 0)>0;pa=int(row.get('PALM_cfwep_voxels','0') or 0)>0
     row['final_status']='PALM_FWER_ACROSS_CONTRASTS' if pa else ('PALM_WITHIN_CONTRAST_ONLY' if pw else ('RANDOMISE_ONLY_NOT_PALM_CONFIRMED' if rs else 'NO_CORRECTED_SPATIAL_EVIDENCE'))
     rows.append(row)
 cols=['model','statistic','contrast_index','contrast_name','final_status','randomise_max_1mp','randomise_voxels','randomise_mm3','PALM_fwep_max_1mp','PALM_fwep_voxels','PALM_fwep_mm3','PALM_cfwep_max_1mp','PALM_cfwep_voxels','PALM_cfwep_mm3','PALM_cfwep_clusters','PALM_fdrp_max_1mp','PALM_fdrp_voxels','randomise_PALM_fwep_dice','randomise_PALM_fwep_map_correlation','PALM_cfwep_peak_row','randomise_corrp_map','PALM_fwep_map','PALM_cfwep_map','screenshot']
 tsv=tabs/'PALM_RANDOMISE_CONTRAST_SUMMARY.tsv'
 with tsv.open('w',newline='') as h:w=csv.DictWriter(h,cols,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows)
 qpath=tabs/'PALM_MODEL_COMPLETION_QC.tsv'
 with qpath.open('w',newline='') as h:
  if qc:w=csv.DictWriter(h,qc[0].keys(),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(qc)
 counts={k:sum(r['final_status']==k for r in rows) for k in ('PALM_FWER_ACROSS_CONTRASTS','PALM_WITHIN_CONTRAST_ONLY','RANDOMISE_ONLY_NOT_PALM_CONFIRMED','NO_CORRECTED_SPATIAL_EVIDENCE')}
 methods=f'PALM was run separately for each finaliser-approved model with {args.nperm:,} permutations, Freedman-Lane regression/permutation, volumetric TFCE, FWER correction within contrasts, joint FWER correction across contrasts within each model, and FDR-adjusted maps. Locked directional contrasts were retained as written. Outputs were saved as 1-p; values above {args.threshold:.2f} indicate corrected p<{1-args.threshold:.2f}. PALM and randomise TFCE values need not be numerically identical because PALM internally harmonises statistics on a z scale for spatial inference and uses a different TFCE discretisation.'
 md=['# Final PALM and randomise permutation report','',f'Generated: {datetime.now().astimezone().isoformat()}','',f'- All models complete: **{complete}**',f'- Models: **{len(models)}**',f'- Tests: **{len(rows)}**']+[f'- {k}: **{v}**' for k,v in counts.items()]+['','## Methods','',methods,'','## Contrast-level results','','| Model | Test | Contrast | Status | randomise voxels | PALM FWER voxels | PALM across-contrast FWER voxels | Dice | Correlation |','|---|---|---|---|---:|---:|---:|---:|---:|']
 for r in rows:md.append(f"| {r['model']} | {r['statistic']} {r['contrast_index']} | {r['contrast_name']} | {r['final_status']} | {r.get('randomise_voxels','')} | {r.get('PALM_fwep_voxels','')} | {r.get('PALM_cfwep_voxels','')} | {r.get('randomise_PALM_fwep_dice','')} | {r.get('randomise_PALM_fwep_map_correlation','')} |")
 md+=['','## Figures','']
 for r in rows:
  if r.get('screenshot') and Path(r['screenshot']).exists():md += [f"### {r['model']}: {r['contrast_name']}",'',f"![map]({os.path.relpath(r['screenshot'],report).replace(os.sep,'/')})",'']
 md += ['## Testing and quality assurance','',f'- Input manifest: `{manifest}`',f'- Completion QC: `{qpath}`',f'- Contrast summary: `{tsv}`','- Input volume count and design row count were validated before PALM.','- Exact commands, seeds, masks, designs and logs are preserved per model.','- Correction across contrasts was within each locked model, not across primary and sensitivity model families.','']
 mdpath=report/'PALM_FINAL_PERMUTATION_REPORT.md';mdpath.write_text('\n'.join(md))
 css='body{font-family:Arial;margin:0;background:#f5f7fa;color:#1f2937}header{background:#17243a;color:#fff;padding:28px 40px}main{max-width:1500px;margin:auto;padding:24px}section{background:#fff;padding:22px;margin:18px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d1d5db;padding:6px;vertical-align:top}th{background:#e5e7eb;position:sticky;top:0}.table-wrap{overflow:auto;max-height:750px}img{max-width:100%}'
 hp=['<!doctype html><html><head><meta charset="utf-8"><title>PALM report</title>',f'<style>{css}</style></head><body><header><h1>Final PALM and randomise permutation report</h1></header><main>',f'<section><h2>Completion</h2><p>All models complete: <strong>{complete}</strong></p><p>Models: {len(models)}; tests: {len(rows)}</p></section>',f'<section><h2>Methods</h2><p>{html.escape(methods)}</p></section>','<section><h2>Contrast-level findings</h2>',html_table(rows,cols[:19]),'</section>']
 imgs=[r for r in rows if r.get('screenshot') and Path(r['screenshot']).exists()]
 if imgs:
  hp.append('<section><h2>Corrected and near-threshold maps</h2>')
  for r in imgs:
   rel=os.path.relpath(r['screenshot'],report).replace(os.sep,'/');hp.append(f"<h3>{html.escape(r['model'])}: {html.escape(r['contrast_name'])}</h3><p>{html.escape(r['final_status'])}</p><img src='{html.escape(rel)}'>")
  hp.append('</section>')
 hp += ['<section><h2>Quality assurance</h2>',html_table(qc,list(qc[0].keys()) if qc else []),'</section></main></body></html>']
 hpath=report/'PALM_FINAL_PERMUTATION_REPORT.html';hpath.write_text(''.join(hp))
 flag=root/('SUCCESS.flag' if complete else 'PARTIAL_REPORT.flag');flag.write_text(f"{'SUCCESS' if complete else 'PARTIAL'}\ncompleted={datetime.now().astimezone().isoformat()}\nreport={hpath}\n")
 print(f'Markdown report: {mdpath}\nHTML report: {hpath}\nContrast table: {tsv}\nAll models complete: {complete}')
 return 0

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 d=sub.add_parser('discover');d.add_argument('--run',type=Path,required=True);d.add_argument('--randomise-root',type=Path);d.add_argument('--output',type=Path,required=True);d.add_argument('--model-token',action='append',default=[])
 r=sub.add_parser('report');r.add_argument('--palm-root',type=Path,required=True);r.add_argument('--nperm',type=int,default=10000);r.add_argument('--threshold',type=float,default=.95);r.add_argument('--near-threshold',type=float,default=.90)
 a=p.parse_args();return discover(a) if a.cmd=='discover' else report(a)
if __name__=='__main__':raise SystemExit(main())
SOURCE_CPSP_PALM_PIPELINE_V1_PY

cat > "$WORK/prepare_submit_cpsp_palm_final_v1.sh" <<'SOURCE_PREPARE_SUBMIT_CPSP_PALM_FINAL_V1_SH'
#!/usr/bin/env bash
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C
PROJECT="${PROJECT:-/mnt/scratch/users/arnastam/fmri_preproc}"
RUN="${RUN:-$PROJECT/derivatives/phase3_stroke/phase5_model_outputs/phase5_models_fixed_primary_secondary_20260707_191818}"
PIPELINE_PY="${PIPELINE_PY:-$PROJECT/cpsp_palm_pipeline_v1.py}"; WORKER="${WORKER:-$PROJECT/run_cpsp_palm_array_v1.sh}"; FINALIZER="${FINALIZER:-$PROJECT/finalize_cpsp_palm_report_v1.sh}"
NPERM="${NPERM:-10000}"; PALM_SEED_BASE="${PALM_SEED_BASE:-2026071400}"; PALM_BIN="${PALM_BIN:-}"; STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"; PALM_ROOT="${PALM_ROOT:-$RUN/PALM_FINAL_$STAMP}"; LOGDIR="${LOGDIR:-$PROJECT/slurm_logs}"
fail(){ echo "ERROR: $1" >&2; exit "${2:-1}"; }
for f in "$PIPELINE_PY" "$WORKER" "$FINALIZER"; do [[ -s "$f" ]] || fail "Missing script: $f"; done; [[ -d "$RUN" ]] || fail "RUN missing: $RUN"
module load fsl >/dev/null 2>&1 || true
if [[ -z "$PALM_BIN" ]]; then module load palm >/dev/null 2>&1 || module load PALM >/dev/null 2>&1 || true; PALM_BIN="$(command -v palm || true)"; fi
if [[ -z "$PALM_BIN" ]]; then for c in "$HOME/software/PALM/palm" "$HOME/PALM/palm" "/mnt/scratch/users/arnastam/software/PALM/palm"; do [[ -x "$c" ]] && PALM_BIN="$c" && break; done; fi
[[ -x "$PALM_BIN" ]] || fail "PALM not found. Run 'module spider palm' or set PALM_BIN=/full/path/to/palm" 10
mkdir -p "$PALM_ROOT/audit" "$PALM_ROOT/logs" "$LOGDIR"; MANIFEST="$PALM_ROOT/audit/PALM_INPUT_MANIFEST.tsv"
python3 "$PIPELINE_PY" discover --run "$RUN" --output "$MANIFEST" --model-token strict_nonCPSP_exclude_NN_DN4pos_minimal --model-token stroke_all_age_sex_adjusted --model-token stroke_all_minimal
N="$(awk 'END{print NR-1}' "$MANIFEST")"; [[ "$N" -eq 3 ]] || fail "Expected 3 models; manifest has $N" 11
cp -p "$PIPELINE_PY" "$WORKER" "$FINALIZER" "$PALM_ROOT/audit/"
printf 'submitted=%s\nproject=%s\nrun=%s\npalm_root=%s\npalm_bin=%s\nnperm=%s\nseed_base=%s\nmodel_count=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "$PROJECT" "$RUN" "$PALM_ROOT" "$PALM_BIN" "$NPERM" "$PALM_SEED_BASE" "$N" > "$PALM_ROOT/audit/SUBMISSION_METADATA.txt"
A="$(sbatch --parsable --job-name=CPSPPALM --partition=nodes --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=2-00:00:00 --array="0-$((N-1))" --output="$LOGDIR/CPSPPALM_%A_%a.out" --error="$LOGDIR/CPSPPALM_%A_%a.err" --export=ALL,PROJECT="$PROJECT",PALM_ROOT="$PALM_ROOT",MANIFEST="$MANIFEST",NPERM="$NPERM",PALM_SEED_BASE="$PALM_SEED_BASE",PALM_BIN="$PALM_BIN" "$WORKER")"
F="$(sbatch --parsable --job-name=PALMReport --partition=nodes --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=08:00:00 --dependency="afterany:$A" --output="$LOGDIR/PALMReport_%j.out" --error="$LOGDIR/PALMReport_%j.err" --export=ALL,PROJECT="$PROJECT",PALM_ROOT="$PALM_ROOT",NPERM="$NPERM",PIPELINE_PY="$PIPELINE_PY" "$FINALIZER")"
printf 'role\tjob_id\tdependency\nPALM_array\t%s\t\nreport_finalizer\t%s\tafterany:%s\n' "$A" "$F" "$A" > "$PALM_ROOT/audit/SLURM_JOBS.tsv"
echo "PALM root: $PALM_ROOT"; echo "PALM array: $A"; echo "Report job: $F"; echo "Monitor: squeue -j $A,$F"; echo "Log: tail -f $LOGDIR/CPSPPALM_${A}_0.out"
SOURCE_PREPARE_SUBMIT_CPSP_PALM_FINAL_V1_SH

cat > "$WORK/run_cpsp_palm_array_v1.sh" <<'SOURCE_RUN_CPSP_PALM_ARRAY_V1_SH'
#!/usr/bin/env bash
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
: "${PALM_ROOT:?}" "${MANIFEST:?}"
NPERM="${NPERM:-10000}"; PALM_SEED_BASE="${PALM_SEED_BASE:-2026071400}"; PALM_BIN="${PALM_BIN:-}"
module load fsl >/dev/null 2>&1 || true
if [[ -z "$PALM_BIN" ]]; then module load palm >/dev/null 2>&1 || module load PALM >/dev/null 2>&1 || true; PALM_BIN="$(command -v palm || true)"; fi
if [[ -z "$PALM_BIN" ]]; then for c in "$HOME/software/PALM/palm" "$HOME/PALM/palm" "/mnt/scratch/users/arnastam/software/PALM/palm"; do [[ -x "$c" ]] && PALM_BIN="$c" && break; done; fi
[[ -x "$PALM_BIN" ]] || { echo "ERROR: PALM executable not found" >&2; exit 10; }
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
eval "$(python3 - "$MANIFEST" "$TASK_ID" <<'PY'
import csv,shlex,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));r=rows[int(sys.argv[2])]
for k in ('model_name','input_4d','mask','design_mat','design_con','design_fts','exchangeability_blocks','variance_groups','npoints','n_t_contrasts','n_f_contrasts'):
 print(f'{k.upper()}={shlex.quote(r.get(k,""))}')
PY
)"
ROOT="$PALM_ROOT/models/$MODEL_NAME"; IN="$ROOT/inputs"; OUT="$ROOT/outputs"; AUD="$ROOT/audit"; LOGD="$ROOT/logs"; mkdir -p "$IN" "$OUT" "$AUD" "$LOGD"
LOG="$LOGD/palm.log"; exec > >(tee "$LOG") 2>&1
fail(){ rc="${2:-1}"; printf 'FAILED\ntime=%s\nmessage=%s\nexit_code=%s\nlog=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "$1" "$rc" "$LOG" > "$ROOT/FAILED.flag"; rm -f "$ROOT/SUCCESS.flag"; echo "ERROR: $1" >&2; exit "$rc"; }
trap 'rc=$?; [[ $rc -eq 0 || -s "$ROOT/FAILED.flag" ]] || fail "Unexpected worker failure" "$rc"' EXIT
for f in "$INPUT_4D" "$MASK" "$DESIGN_MAT" "$DESIGN_CON"; do [[ -s "$f" ]] || fail "Missing input: $f" 11; done
D4="$(fslval "$INPUT_4D" dim4 | awk '{print int($1)}')"; [[ "$D4" == "$NPOINTS" ]] || fail "dim4=$D4 but design NumPoints=$NPOINTS" 12
MV="$(fslstats "$MASK" -V | awk '{print int($1)}')"; [[ "$MV" -gt 0 ]] || fail "Mask has zero voxels" 13
copy_nii(){ if [[ "$1" == *.nii.gz ]]; then gzip -dc "$1" > "$2"; elif [[ "$1" == *.nii ]]; then cp -p "$1" "$2"; else fail "Unsupported NIfTI: $1" 14; fi; }
copy_nii "$INPUT_4D" "$IN/input_4d.nii"; copy_nii "$MASK" "$IN/mask.nii"; cp -p "$DESIGN_MAT" "$IN/design.mat"; cp -p "$DESIGN_CON" "$IN/design.con"; [[ -n "$DESIGN_FTS" && -s "$DESIGN_FTS" ]] && cp -p "$DESIGN_FTS" "$IN/design.fts"
SEED=$((PALM_SEED_BASE+TASK_ID+1)); CMD=("$PALM_BIN" -i "$IN/input_4d.nii" -m "$IN/mask.nii" -d "$IN/design.mat" -t "$IN/design.con")
[[ -s "$IN/design.fts" ]] && CMD+=( -f "$IN/design.fts" )
[[ -n "$EXCHANGEABILITY_BLOCKS" && -s "$EXCHANGEABILITY_BLOCKS" ]] && CMD+=( -eb "$EXCHANGEABILITY_BLOCKS" )
[[ -n "$VARIANCE_GROUPS" && -s "$VARIANCE_GROUPS" ]] && CMD+=( -vg "$VARIANCE_GROUPS" )
CMD+=( -n "$NPERM" -T -corrcon -fdr -save1-p -savemetrics -savemax -savedof -savemask -rmethod Freedman-Lane -seed "$SEED" -o "$OUT/palm" )
[[ "${PALM_TWO_TAILED:-0}" == 1 ]] && CMD+=( -twotail )
{ echo '#!/usr/bin/env bash'; printf '%q ' "${CMD[@]}"; echo; } > "$AUD/PALM_COMMAND.sh"; chmod +x "$AUD/PALM_COMMAND.sh"
printf 'model=%s\ntask_id=%s\nnperm=%s\nseed=%s\ndim4=%s\nmask_voxels=%s\nn_t_contrasts=%s\nn_f_contrasts=%s\ntwo_tailed=%s\npalm_bin=%s\nstarted=%s\nhost=%s\n' "$MODEL_NAME" "$TASK_ID" "$NPERM" "$SEED" "$D4" "$MV" "$N_T_CONTRASTS" "$N_F_CONTRASTS" "${PALM_TWO_TAILED:-0}" "$PALM_BIN" "$(date --iso-8601=seconds 2>/dev/null || date)" "$(hostname)" > "$AUD/PALM_RUN_METADATA.txt"
echo "Running PALM: $MODEL_NAME"; printf 'Command: '; printf '%q ' "${CMD[@]}"; echo; "${CMD[@]}"
EXPECTED=$((N_T_CONTRASTS+N_F_CONTRASTS)); C="$(find "$OUT" -maxdepth 1 -type f -name '*_tfce_*stat_cfwep*.nii*' | wc -l)"; F="$(find "$OUT" -maxdepth 1 -type f -name '*_tfce_*stat_fwep*.nii*' | wc -l)"
[[ "$C" -ge "$EXPECTED" ]] || fail "Expected >=$EXPECTED cfwep maps; found $C" 18; [[ "$F" -ge "$EXPECTED" ]] || fail "Expected >=$EXPECTED fwep maps; found $F" 19
printf 'SUCCESS\ncompleted=%s\nmodel=%s\ncfwep_maps=%s\nfwep_maps=%s\nlog=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "$MODEL_NAME" "$C" "$F" "$LOG" > "$ROOT/SUCCESS.flag"; rm -f "$ROOT/FAILED.flag"
SOURCE_RUN_CPSP_PALM_ARRAY_V1_SH

cat > "$WORK/finalize_cpsp_palm_report_v1.sh" <<'SOURCE_FINALIZE_CPSP_PALM_REPORT_V1_SH'
#!/usr/bin/env bash
set -Eeuo pipefail
umask 0027
export LC_ALL=C LANG=C
: "${PALM_ROOT:?}" "${PROJECT:?}"
NPERM="${NPERM:-10000}"; PIPELINE_PY="${PIPELINE_PY:-$PROJECT/cpsp_palm_pipeline_v1.py}"
module load fsl >/dev/null 2>&1 || true
mkdir -p "$PALM_ROOT/logs"; exec > >(tee "$PALM_ROOT/logs/finalize_report.log") 2>&1
python3 "$PIPELINE_PY" report --palm-root "$PALM_ROOT" --nperm "$NPERM" --threshold .95 --near-threshold .90
find "$PALM_ROOT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > "$PALM_ROOT/audit/SHA256SUMS.txt"
ARCHIVE="${PALM_ROOT}.tar.gz"; tar -czf "$ARCHIVE" -C "$(dirname "$PALM_ROOT")" "$(basename "$PALM_ROOT")"
echo "HTML: $PALM_ROOT/report/PALM_FINAL_PERMUTATION_REPORT.html"; echo "Markdown: $PALM_ROOT/report/PALM_FINAL_PERMUTATION_REPORT.md"; echo "Archive: $ARCHIVE"
SOURCE_FINALIZE_CPSP_PALM_REPORT_V1_SH

chmod u+x "$WORK"/*.sh "$WORK"/*.py 2>/dev/null || true

primary_analysis() {
    # LNM and BCB analysis
    submit_script prepare_heat_only_all_branches.sh
    run_script validate_bundle.py
    run_script submit_cpsp_pc2_parallel.sh
}

localisation_analysis() {
    # Regional localisation
    submit_script build_morel_posterior_sensory_targets_v1.sh
    submit_script run_thalamic_operculo_insular_S2_subregions_LNM_BCB_four_outcomes_v1_1_FIXED.sh
}

robustness_analysis() {
    # Sensitivity analyses
    local script="${THALAMIC_ROBUSTNESS_SCRIPT:-}"

    if [[ -z "$script" ]]; then
        script="$(
            find /mnt/scratch/users/arnastam /users/arnastam/fmri_preproc \
                -type f -name run_thalamic_headline_robustness_v1.sh \
                2>/dev/null | head -1 || true
        )"
    fi

    [[ -n "$script" && -s "$script" ]] || {
        echo "Set THALAMIC_ROBUSTNESS_SCRIPT to run_thalamic_headline_robustness_v1.sh." >&2
        exit 3
    }

    if command -v sbatch >/dev/null 2>&1; then
        sbatch --wait "$script"
    else
        bash "$script"
    fi
}

spatial_analysis() {
    # Whole-brain spatial analysis
    PREPARE_SCRIPT="$WORK/prepare_partB_randomise_bulletproof_v1_1.sh" \
    RUN_SCRIPT="$WORK/run_one_partB_lnm_randomise_v1.sh" \
    FINALISE_SCRIPT="$WORK/finalise_partB_randomise_bulletproof_v1_1.sh" \
        run_script submit_partB_randomise_bulletproof_v1_1.sh

    PROJECT=/mnt/scratch/users/arnastam/fmri_preproc \
    PIPELINE_PY="$WORK/cpsp_palm_pipeline_v1.py" \
    WORKER="$WORK/run_cpsp_palm_array_v1.sh" \
    FINALIZER="$WORK/finalize_cpsp_palm_report_v1.sh" \
        run_script prepare_submit_cpsp_palm_final_v1.sh
}

case "$MODE" in
    primary)      primary_analysis ;;
    localisation) localisation_analysis ;;
    robustness)   robustness_analysis ;;
    spatial)      spatial_analysis ;;
    -h|--help|help)
        echo "Usage: $0 [primary|localisation|robustness|spatial]"
        ;;
    *)
        echo "Unknown analysis stage: $MODE" >&2
        echo "Usage: $0 [primary|localisation|robustness|spatial]" >&2
        exit 2
        ;;
esac
