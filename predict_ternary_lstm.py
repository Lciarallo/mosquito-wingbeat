"""Three-class audio inference from a frozen LSTM/Bi-LSTM with optional MC dropout."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from train_ternary_lstm import frontend, GroupPool, read_json
from ternary_protocol import CLASS_NAMES, read_audio, apply_policy
from torres_lstm import TorresLSTM, digest, predict_probabilities, monte_carlo_probabilities, write_json


def predict(run, audio, *, name=None, passes=100, device="auto"):
    run = Path(run)
    completed = read_json(run / "complete.json")
    if not completed["complete"]:
        raise ValueError("The experiment has not finished.")
    root = read_json(run / "freeze.json")
    name = name or root["selected_final_neural_model"]
    directory = run / "final" / name
    frozen = read_json(directory / "freeze.json")
    if digest(directory / "freeze.json") != root["context_freeze_sha256"][f"final/{name}"]:
        raise ValueError("Final freeze has changed.")
    if digest(directory / "model.pt") != frozen["checkpoint_sha256"] or digest(directory / "scaler.npz") != frozen["scaler_sha256"]:
        raise ValueError("Frozen model/scaler verification failed.")
    device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model = TorresLSTM(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    wave = read_audio(Path(audio), 8000)
    n = len(wave) // 7936
    if not n:
        raise ValueError("At least 0.992 s of real audio is required; no padding or repetition.")
    features = frontend().to(device).eval()
    with torch.inference_mode():
        x = np.concatenate([features(torch.tensor(wave[s*7936:min(s+256,n)*7936].reshape(-1,7936).copy(),device=device)).cpu().numpy()
                            for s in range(0,n,256)])
    with np.load(directory / "scaler.npz", allow_pickle=False) as scaler:
        x = ((x - scaler["mean"]) / scaler["std"]).astype("float32")
    frame = pd.DataFrame(dict(group=["audio"]*n,true_label=0,pcm_block="audio",domain="inference"))
    pool = GroupPool(frame)
    p = predict_probabilities(model,x,device)
    group = pool.table(p)
    decision = int(apply_policy(group,frozen["policy"])[0])
    result = dict(model=name, decision=CLASS_NAMES[decision], predicted_class=decision,
        class_order=CLASS_NAMES, deterministic_probabilities=pool.probabilities(p)[0].tolist(),
        windows=n, real_seconds=len(wave)/8000, ignored_tail_seconds=(len(wave)-n*7936)/8000,
        frontend=checkpoint["frontend"], decision_rule="frozen calibrated cascade on activity-pooled probabilities")
    if passes:
        draws = monte_carlo_probabilities(model,x,device,passes=passes)
        labels = np.array([int(apply_policy(pool.table(draw),frozen["policy"])[0]) for draw in draws])
        votes = np.bincount(labels,minlength=3)
        result.update(mc_passes=passes,mc_majority_decision=CLASS_NAMES[int(votes.argmax())],
            mc_votes={c:int(v) for c,v in zip(CLASS_NAMES,votes)},mc_disagreement=float(1-votes.max()/passes))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio",type=Path)
    parser.add_argument("--run",type=Path,default=Path(__file__).resolve().parent/"results/ternary_lstm_2026_10_15")
    parser.add_argument("--model",choices=["lstm","bilstm"])
    parser.add_argument("--mc-passes",type=int,default=100)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Existing inference output is preserved.")
    if args.mc_passes not in [0] and args.mc_passes<2:
        parser.error("Use zero or at least two Monte Carlo passes.")
    torch.set_num_threads(4)
    result=predict(args.run,args.audio,name=args.model,passes=args.mc_passes,device=args.device)
    if args.output:
        write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
