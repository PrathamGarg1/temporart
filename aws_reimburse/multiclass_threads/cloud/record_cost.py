#!/usr/bin/env python3
"""Compute the EC2 run's cost from AWS facts and append it to the reimbursement ledger.

  python cloud/record_cost.py            # after the instance is terminated

Evidence used (all from AWS, not estimated by us):
  * instance LaunchTime and termination time (describe-instances / StateTransitionReason)
  * on-demand hourly price from the AWS Price List API
  * EBS gp3 price from the Price List API; S3 charged at list price on bytes stored
Cross-check later with Cost Explorer filtered by tag Project=surakshanet-multiclass-threads.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from cost_ledger import log_call  # noqa: E402

LOC = "US East (N. Virginia)"


def aws(*args: str) -> str:
    return subprocess.run(["aws", *args], check=True, capture_output=True, text=True).stdout


def price(service: str, filters: list[tuple[str, str]], unit_hint: str = "") -> float:
    # one --filters flag only: repeating it makes the CLI keep just the last filter
    f = ["--filters", *[f"Type=TERM_MATCH,Field={k},Value={v}" for k, v in filters]]
    out = aws("pricing", "get-products", "--region", "us-east-1", "--service-code", service, *f)
    items = json.loads(out)["PriceList"]
    if len(items) != 1:
        raise SystemExit(f"price lookup ambiguous: {len(items)} products for {filters}")
    d = json.loads(items[0])
    dims = list(list(d["terms"]["OnDemand"].values())[0]["priceDimensions"].values())
    return float(dims[0]["pricePerUnit"]["USD"])


def parse_ts(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def main() -> None:
    st = json.loads((HERE / "run_state.json").read_text())
    region, iid, itype, bucket = st["region"], st["instance_id"], st["instance_type"], st["bucket"]

    d = json.loads(aws("ec2", "describe-instances", "--region", region, "--instance-ids", iid))
    inst = d["Reservations"][0]["Instances"][0]
    state = inst["State"]["Name"]
    launch = parse_ts(inst["LaunchTime"])
    end = None
    m = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)", inst.get("StateTransitionReason", ""))
    if m:
        end = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if state != "terminated":
        raise SystemExit(f"instance is {state}; run again after it terminates (bills until then)")
    if end is None:
        meta = json.loads(aws("s3", "cp", f"s3://{bucket}/progress/run_meta.json", "-", "--region", region))
        end = datetime.fromtimestamp(meta["end_epoch"] + 30, tz=timezone.utc)
        print("end time from run_meta (+30s shutdown margin)")

    secs = max((end - launch).total_seconds(), 60)  # 60 s minimum billing
    hours = secs / 3600

    p_inst = price(
        "AmazonEC2",
        [("instanceType", itype), ("location", LOC), ("operatingSystem", "Linux"), ("tenancy", "Shared"),
         ("preInstalledSw", "NA"), ("capacitystatus", "Used")],
    )
    p_ebs = price("AmazonEC2", [("volumeApiName", "gp3"), ("location", LOC), ("productFamily", "Storage")])
    ec2_usd = p_inst * hours
    ebs_usd = p_ebs * st["ebs_gb"] * hours / 730.0

    out = aws("s3", "ls", f"s3://{bucket}/", "--recursive", "--summarize", "--region", region)
    gb = int(re.search(r"Total Size: (\d+)", out).group(1)) / 1e9
    s3_usd = gb * 0.023 * (hours / 730.0)

    note = (
        f"{iid} {itype} launch={launch.isoformat()} end={end.isoformat()} "
        f"hours={hours:.4f} price={p_inst}/h; tags Project=surakshanet-multiclass-threads"
    )
    log_call(job="multiclass_train_v2_gpu", service="ec2", model=itype, input_tokens=0, output_tokens=0,
             notes=note, usd_override=ec2_usd)
    log_call(job="multiclass_train_v2_gpu", service="ebs_gp3", model=f"{st['ebs_gb']}GB", input_tokens=0,
             output_tokens=0, notes=f"{p_ebs}/GB-month x {st['ebs_gb']} GB x {hours:.4f}h", usd_override=ebs_usd)
    log_call(job="multiclass_train_v2_gpu", service="s3", model="standard", input_tokens=0, output_tokens=0,
             notes=f"{gb:.2f} GB for {hours:.2f}h (upper bound)", usd_override=s3_usd)

    summary = {
        "instance_id": iid, "instance_type": itype, "launch_utc": launch.isoformat(), "end_utc": end.isoformat(),
        "hours": round(hours, 4), "usd_ec2": round(ec2_usd, 4), "usd_ebs": round(ebs_usd, 4),
        "usd_s3": round(s3_usd, 4), "usd_total": round(ec2_usd + ebs_usd + s3_usd, 4),
    }
    (HERE / "run_cost.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
