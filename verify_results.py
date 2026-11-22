#!/usr/bin/env python3
"""Compare reproduced numerical output with the repository's recorded result."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def differences(expected, actual, path="root", atol=1e-8):
    """Recursively report mismatches; environment metadata is not compared."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict):
            return [f"{path}: expected an object"]
        diffs=[]
        for key,value in expected.items():
            if path=="root" and key=="environment":
                continue
            if key not in actual:
                diffs.append(f"{path}.{key}: missing")
            else:
                diffs.extend(differences(value,actual[key],f"{path}.{key}",atol))
        for key in actual.keys()-expected.keys():
            diffs.append(f"{path}.{key}: unexpected")
        return diffs
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):
            return [f"{path}: list length or type differs"]
        return [difference for i,(left,right) in enumerate(zip(expected,actual))
                for difference in differences(left,right,f"{path}[{i}]",atol)]
    if isinstance(expected,float):
        if not isinstance(actual,(int,float)) or isinstance(actual,bool) or not math.isclose(
                expected,float(actual),rel_tol=atol,abs_tol=atol):
            return [f"{path}: {actual!r} differs from {expected!r}"]
        return []
    return [] if type(expected) is type(actual) and expected==actual else [
        f"{path}: {actual!r} differs from {expected!r}"
    ]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",type=Path,
                        help="Directory containing reproduced original results.json")
    parser.add_argument("--train-cv",type=Path,
                        help="Directory containing reproduced train CV summary.json")
    parser.add_argument("--blur",type=Path,
                        help="Optional directory containing reproduced blur sensitivity summary.json")
    parser.add_argument("--pretrained",type=Path,
                        help="Optional directory containing reproduced ResNet comparison summary.json")
    parser.add_argument("--woundsdb",type=Path,
                        help="Optional directory containing reproduced WoundsDB audit summary.json")
    parser.add_argument("--calibration",type=Path,
                        help="Optional directory containing reproduced calibration audit summary.json")
    parser.add_argument("--split-sensitivity",type=Path,
                        help="Optional directory containing reproduced split sensitivity summary.json")
    parser.add_argument("--patient-pilot",type=Path,
                        help="Optional directory containing reproduced LUTSeg patient-grouped pilot summary.json")
    parser.add_argument("--mask-audit",type=Path,
                        help="Optional directory containing reproduced LUTSeg mask audit summary.json")
    parser.add_argument("--atol",type=float,default=1e-8,
                        help="Absolute and relative float tolerance, default 1e-8")
    args=parser.parse_args()
    if not math.isfinite(args.atol) or args.atol<0:
        parser.error("--atol must be non-negative and finite")
    root=Path(__file__).resolve().parent
    pairs=[]
    if args.original is not None:
        pairs.append((root/"results/original/results.json",args.original/"results.json"))
    if args.train_cv is not None:
        pairs.append((root/"results/train_cv/summary.json",args.train_cv/"summary.json"))
    if args.blur is not None:
        pairs.append((root/"results/blur_sensitivity/summary.json",args.blur/"summary.json"))
    if args.pretrained is not None:
        pairs.append((root/"results/pretrained_comparison/summary.json",args.pretrained/"summary.json"))
    if args.woundsdb is not None:
        pairs.append((root/"results/woundsdb_audit/summary.json",args.woundsdb/"summary.json"))
    if args.calibration is not None:
        pairs.append((root/"results/calibration_audit/summary.json",args.calibration/"summary.json"))
    if args.split_sensitivity is not None:
        pairs.append((root/"results/split_sensitivity/summary.json",args.split_sensitivity/"summary.json"))
    if args.patient_pilot is not None:
        pairs.append((root/"results/lutseg_patient_pilot/summary.json",args.patient_pilot/"summary.json"))
    if args.mask_audit is not None:
        pairs.append((root/"results/lutseg_mask_audit/summary.json",args.mask_audit/"summary.json"))
    if not pairs:
        parser.error("provide at least one results directory to verify")
    issues=[]
    for reference,reproduced in pairs:
        expected=json.loads(reference.read_text(encoding="utf-8"))
        actual=json.loads(reproduced.read_text(encoding="utf-8"))
        issues.extend(differences(expected,actual,path="root",atol=args.atol))
    if issues:
        print("Reproduction differences found:")
        for issue in issues[:50]:
            print("-",issue)
        if len(issues)>50:
            print(f"... {len(issues)-50} more")
        raise SystemExit(1)
    print(f"All {len(pairs)} requested outputs match the recorded results.")


if __name__=="__main__":
    main()
