"""Smoke-test the wrapper bits without spinning up a full XTTS run."""
import sys, threading, math
sys.path.insert(0, '/home/ubspace/Desktop/voiceforge')

from backend.pipelines.training import ProgressBridge, _peek_sample_rate
from backend.pipelines.training_config import _build_plan, Preset


# ── 1. Bridge: monotonic + cancel ─────────────────────────────────
class FakeTrainer:
    def __init__(self, total_steps_planned, batches_per_epoch):
        self.total_steps_done = 0
        self.epochs_done = 0
        class _L:
            def __init__(self, n): self.n = n
            def __len__(self): return self.n
        self.train_loader = _L(batches_per_epoch)


calls = []
def capture(job_id, percent, message):
    calls.append((job_id, percent, message))


ft = FakeTrainer(total_steps_planned=60, batches_per_epoch=10)
bridge = ProgressBridge(job_id="J", total_epochs=6, update_fn=capture)
bridge.on_init_end(ft)
for epoch in range(6):
    bridge.on_epoch_start(ft)
    for s in range(10):
        ft.total_steps_done += 1
        bridge.on_train_step_end(ft)
    ft.epochs_done = epoch + 1
    bridge.on_epoch_end(ft)

percents = [c[1] for c in calls]
assert percents == sorted(percents), f"non-monotonic: {percents}"
assert max(percents) <= 99, "should never exceed 99 mid-run"
print(f"✓ {len(calls)} updates, monotonic 0..{max(percents)}, cap respected")

# Cancellation
ev = threading.Event()
ft2 = FakeTrainer(total_steps_planned=60, batches_per_epoch=10)
b2 = ProgressBridge(job_id="J2", total_epochs=6, cancel_event=ev, update_fn=lambda *a: None)
b2.on_epoch_start(ft2)
ft2.total_steps_done = 5
b2.on_train_step_end(ft2)
ev.set()
try:
    b2.on_train_step_end(ft2)
    print("✗ Expected KeyboardInterrupt"); sys.exit(1)
except KeyboardInterrupt:
    print("✓ cancel_event raises KeyboardInterrupt cleanly")


# ── 2. Plan: target_steps drives epochs sensibly ──────────────────
# 28 train clips, batch=2, accum=4 → 14 micro-batches/epoch / 4 = 3 updates/epoch
# At target_steps=250, we need ceil(250/3) = 84 epochs. That's the user's
# entire complaint — and it's now visible at config time, not by surprise.
plan = _build_plan(Preset.LOW_VRAM, vram_gb=4.0)
assert plan.target_steps == 250
assert plan.epochs == 0  # filled in by training.py
print(f"✓ low-vram plan: target_steps={plan.target_steps}, batch={plan.batch_size}, accum={plan.grad_accum_steps}")

# Manually do the calculation training.py will do:
n_train = 28
mb_per_epoch = max(1, n_train // plan.batch_size)
upd_per_epoch = max(1, mb_per_epoch // max(1, plan.grad_accum_steps))
epochs_needed = max(1, math.ceil(plan.target_steps / upd_per_epoch))
print(f"  → on {n_train} clips: {upd_per_epoch} updates/epoch, {epochs_needed} epochs to hit target")
assert epochs_needed > 6, "with 28 clips we *must* need more than 6 epochs to hit 250 updates"

# And on a healthier 200-clip dataset we should land near a normal epoch count
n_train = 200
mb_per_epoch = max(1, n_train // plan.batch_size)
upd_per_epoch = max(1, mb_per_epoch // max(1, plan.grad_accum_steps))
epochs_needed = max(1, math.ceil(plan.target_steps / upd_per_epoch))
print(f"  → on 200 clips: {upd_per_epoch} updates/epoch, {epochs_needed} epochs")
assert 5 <= epochs_needed <= 15, f"unreasonable epoch count for big dataset: {epochs_needed}"

print("\n✓ all smoke checks passed")
