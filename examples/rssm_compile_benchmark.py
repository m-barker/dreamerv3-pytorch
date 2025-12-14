import time
import torch

from dreamer.networks.rssm import RSSM


class RSSMNoScanWrapper(torch.nn.Module):
    def __init__(self, rssm):
        super().__init__()
        self.rssm = rssm

    def forward(self, prev_actions, encoded_obs, is_first_mask):
        return self.rssm.observe_sequence_no_scan(
            prev_actions, encoded_obs, is_first_mask
        )


# ---------------------------
# Timing helpers
# ---------------------------


def timed(fn, *, sync=True):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, out


def benchmark(name, fn, n_runs=10):
    times = []
    for _ in range(n_runs):
        dt, _ = timed(fn)
        times.append(dt)
    avg = sum(times) / len(times)
    print(f"{name:<35} {avg:.6f}s")
    return avg


# ---------------------------
# Benchmark
# ---------------------------


def benchmark_rssm(
    rssm,
    prev_actions,
    encoded_obs,
    is_first_mask,
    *,
    use_amp=False,
    compile_model=False,
    n_warmup=3,
    n_runs=10,
):
    model = RSSMNoScanWrapper(rssm).eval()

    if compile_model:
        model = torch.compile(
            model,
            mode="default",
            fullgraph=False,
        )

    def forward():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(prev_actions, encoded_obs, is_first_mask)
        else:
            return model(prev_actions, encoded_obs, is_first_mask)

    # ---------------------------
    # Warmup (includes compile)
    # ---------------------------
    for _ in range(n_warmup):
        _ = forward()

    torch.cuda.synchronize()

    # ---------------------------
    # Timed runs
    # ---------------------------
    return benchmark(
        f"{'compile' if compile_model else 'eager'}{' + amp' if use_amp else ''}",
        forward,
        n_runs=n_runs,
    )


# ---------------------------
# Main
# ---------------------------


def main():
    device = torch.device("cuda")
    print("Using device:", device)

    # ---------------------------
    # Model / data config
    # ---------------------------
    batch_size = 16
    seq_length = 64
    deter_dim = 128
    n_stoch_dist = 16
    n_stoch_cats = 16
    embed_size = 64
    action_dim = 10

    torch.manual_seed(0)

    rssm = RSSM(
        deter_size=deter_dim,
        n_stoch_dists=n_stoch_dist,
        n_stoch_cats=n_stoch_cats,
        encoded_size=embed_size,
        hidden_size=64,
        act_func="ReLU",
        n_prior_layers=1,
        n_post_layers=2,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        action_dim=action_dim,
        device=device,
    )

    prev_actions = torch.randn(batch_size, seq_length, action_dim, device=device)
    encoded_obs = torch.randn(batch_size, seq_length, embed_size, device=device)
    is_first_mask = torch.zeros(
        batch_size, seq_length, 1, device=device, dtype=torch.int32
    )

    print("\n=== RSSM observe_sequence benchmark ===\n")

    # ---------------------------
    # Run benchmarks
    # ---------------------------

    benchmark_rssm(
        rssm,
        prev_actions,
        encoded_obs,
        is_first_mask,
        use_amp=False,
        compile_model=False,
    )

    benchmark_rssm(
        rssm,
        prev_actions,
        encoded_obs,
        is_first_mask,
        use_amp=True,
        compile_model=False,
    )

    benchmark_rssm(
        rssm,
        prev_actions,
        encoded_obs,
        is_first_mask,
        use_amp=False,
        compile_model=True,
    )

    benchmark_rssm(
        rssm,
        prev_actions,
        encoded_obs,
        is_first_mask,
        use_amp=True,
        compile_model=True,
    )


if __name__ == "__main__":
    main()
