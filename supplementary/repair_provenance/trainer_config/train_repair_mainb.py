"""
Main-line B two-stream path-B repair trainer (V1 8L student).

This file does NOT replace distill_oxe.py. It reuses distill_oxe.py's KD+action loss
verbatim (OXEDistillationTrainer.make_loss_fn) and teacher_pipeline.py's verified
make_loader as-is (red lines 3/5). Locked recipe — do not change the design here.

Locked config (see spec):
  - start  : student = OctoModel.load_pretrained(V1 300k), already 8L, assert num_layers==8.
             V1 dir is READ-ONLY (red line 2). teacher = octo-base-1.5 (12L, no-grad).
  - opt    : fresh TrainState (V1 params + new tx); Adam moments reset to zero.
  - LR     : flat 1e-6, warmup 200 steps, no decay (create_lr_schedule "constant").
  - train  : all student params trainable (no freeze, no stop_gradient on student).
  - batch  : ONE combined batch of 128 = preserve(B_P=102) ++ repair(B_R=26) along axis 0;
             one student fwd + one teacher fwd + one backward. dose = B_R/128 = rho_R.
  - loss   : total = 0.5*KD + 0.5*action over the full 128 (standard mean reduction).
  - save   : every 2000 steps to a NEW dir; never writes checkpoints_distill_v2 (red line 2).

THIS ROUND: smoke-train only (--smoke). Never run the full job (red line 9).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# HF cache on the persistent volume (red line 11: downloading assets is allowed; this
# does not modify the conda env). t5-base + octo-base-1.5 are already cached here.
os.environ.setdefault("HF_HOME", ".cache/huggingface")

import sys
from functools import partial

import numpy as np
from absl import app, flags, logging
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import wandb

from octo.model.octo_model import OctoModel
from octo.data.dataset import make_single_dataset, make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_utils import create_optimizer, process_text, TrainState
from octo.utils.train_callbacks import SaveCallback

# --- verbatim reuse of V1 distillation KD+action loss (red line 3: do not modify it) ---
sys.path.insert(0, ".")
from distill_oxe import OXEDistillationTrainer

# Importing distill_oxe registers its module-level absl flags (name, teacher_checkpoint,
# save_dir=checkpoints_distill_v2, ...). We only want the class, not its CLI — drop those
# flags so THIS file's flags (with safe defaults) are authoritative and don't DuplicateFlagError.
# This also removes distill_oxe's dangerous save_dir default pointing at the read-only V1 dir.
for _n in [
    "name", "debug", "teacher_checkpoint", "save_dir", "data_dir", "data_mix",
    "batch_size", "student_layers", "alpha", "num_steps", "wandb_project", "wandb_entity",
]:
    if _n in flags.FLAGS:
        delattr(flags.FLAGS, _n)

# --- verified repair loader, reused as-is (red line 5: do not re-parameterize internals) ---
sys.path.insert(0, "teacher_data_code")
import sim_eggplant_teacher_dataset_builder  # noqa: F401  (registers the TFDS builder)
from teacher_pipeline import make_loader


FLAGS = flags.FLAGS

flags.DEFINE_bool("smoke", False, "Smoke-train: ~10 steps, no save/wandb, print diagnostics.")
flags.DEFINE_bool("full", False, "Full run: num_steps, wandb, checkpoint every save_interval.")
flags.DEFINE_string("name", "mainb_repair_rhoR020_predB", "Run name / save subdir.")
flags.DEFINE_string("wandb_project", "octo_repair_mainb", "wandb project.")
flags.DEFINE_string("v1_path", "checkpoints_distill_v2", "V1 checkpoint root (READ-ONLY).")
flags.DEFINE_integer("v1_step", 300000, "V1 step to fork from.")
flags.DEFINE_string("teacher_checkpoint", "hf://rail-berkeley/octo-base-1.5", "12L teacher.")
flags.DEFINE_string("preserve_data_dir", "downloads", "OXE preserve-stream data root.")
flags.DEFINE_string("preserve_data_mix", "oxe_magic_soup", "OXE mix name.")
flags.DEFINE_string("repair_data_dir", "tfds_data", "SimEggplantTeacher repair-stream root.")
flags.DEFINE_string("save_dir", "v2_runs/mainb_repair_rhoR020_predB", "NEW save dir (never V1).")

flags.DEFINE_integer("batch_size", 128, "Combined batch size (== V1 footprint).")
flags.DEFINE_float("rho_R", 0.2, "Repair fraction of the batch (dose). Ladder: {0.2,0.35,0.5}.")
flags.DEFINE_float("alpha", 0.5, "KD weight; action weight = 1 - alpha.")
flags.DEFINE_integer("num_steps", 10000, "Full-run target (NOT run this round).")
flags.DEFINE_integer("save_interval", 2000, "Checkpoint interval (full run only).")
flags.DEFINE_integer("seed", 42, "RNG seed.")
flags.DEFINE_integer("smoke_steps", 10, "Number of steps for --smoke.")

# locked LR (flat, short warmup, no decay)
LR_PEAK = 1e-6
LR_WARMUP = 200
WEIGHT_DECAY = 0.01
CLIP_GRADIENT = 1.0
REPAIR_SHUFFLE_BUFFER = 2048


# =============================================================================
# Host-side batch alignment (concat preserve ++ repair along axis 0)
# =============================================================================
def tree_concat(a, b, dropped, path=""):
    """Recursively concat two nested numpy batches along axis 0 over the shared keys.
    Non-shared keys are dropped (and recorded). Trailing shapes must match or we raise."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a.keys()), set(b.keys())
        for k in sorted((ka | kb) - (ka & kb)):
            dropped.append(f"{path}/{k}")
        out = {}
        for k in sorted(ka & kb):
            out[k] = tree_concat(a[k], b[k], dropped, f"{path}/{k}")
        return out
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(
            f"Structural mismatch at '{path}': preserve {a.shape} vs repair {b.shape} "
            f"(trailing dims differ; cannot clean-concat — STOP and report)."
        )
    return np.concatenate([a, b], axis=0)


def tree_slice(tree, sl):
    if isinstance(tree, dict):
        return {k: tree_slice(v, sl) for k, v in tree.items()}
    return tree[sl]


def tree_struct(tree, prefix=""):
    """Flat {path: (shape, dtype)} for printing live batch structure."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(tree_struct(v, f"{prefix}/{k}"))
        return out
    arr = np.asarray(tree)
    out[prefix] = (tuple(arr.shape), str(arr.dtype))
    return out


def _print_struct(name, batch):
    logging.info(f"--- {name} batch structure ---")
    for path, (shape, dtype) in tree_struct(batch).items():
        logging.info(f"    {path:55s} {str(shape):24s} {dtype}")


# =============================================================================
# Stream builders
# =============================================================================
def build_preserve_iter(B_P, text_processor):
    """OXE preserve stream — distill_oxe.py pipeline reused verbatim, only batch_size -> B_P."""
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        FLAGS.preserve_data_mix,
        data_dir=FLAGS.preserve_data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,
        load_language=True,
        action_proprio_normalization_type="normal",
    )
    ds = make_interleaved_dataset(
        dataset_kwargs_list=dataset_kwargs_list,
        sample_weights=sample_weights,
        train=True,
        shuffle_buffer_size=50000,
        traj_transform_kwargs=dict(window_size=2, action_horizon=4),
        frame_transform_kwargs=dict(resize_size={"primary": (256, 256)}),
        batch_size=B_P,
        balance_weights=True,
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    def gen():
        for batch in ds.iterator():
            batch = process_text(batch, text_processor)
            batch.pop("dataset_name", None)
            yield batch

    return gen()


def build_repair_iter(B_R, text_processor):
    """Path-B repair stream: make_loader as-is (train=True 95% + train=False 5%) to cover all 324.
    Reuses the verified loader unchanged (red line 5); train/val are combined at the dataset level."""
    os.environ["CHUNK_SOURCE"] = "predicted"  # path B (BC target = teacher-predicted chunk)
    ds_train = make_loader(FLAGS.repair_data_dir, train=True, shuffle=True)   # train[:95%]
    ds_val = make_loader(FLAGS.repair_data_dir, train=False, shuffle=True)    # train[95%:]
    # flatten each (DLataset -> frames) BEFORE leaving the DLataset type, then concat+repeat+shuffle+batch
    frames = ds_train.flatten().concatenate(ds_val.flatten())
    frames = frames.repeat().shuffle(REPAIR_SHUFFLE_BUFFER, seed=FLAGS.seed).batch(B_R)

    def gen():
        for batch in frames.as_numpy_iterator():
            batch = process_text(batch, text_processor)
            yield batch

    return gen()


def count_repair_coverage():
    """One-time count of repair episodes AND exact windowed-frame count, to confirm full 324
    coverage (not 308) and to pin the epoch denominator. Frames = sum of per-traj action.shape[0]
    (== number of elements after .flatten()); counted in the same pass to avoid a second decode."""
    try:
        import tensorflow_datasets as tfds
        total = tfds.builder("sim_eggplant_teacher", data_dir=FLAGS.repair_data_dir).info.splits["train"].num_examples
        logging.info(f"TFDS builder reports train split num_examples = {total}")
    except Exception as e:
        total = None
        logging.warning(f"Could not read TFDS builder info: {e}")

    os.environ["CHUNK_SOURCE"] = "predicted"

    def _count(split_train):
        ne = nf = 0
        for traj in make_loader(FLAGS.repair_data_dir, train=split_train, shuffle=False).iterator():
            ne += 1
            nf += int(np.asarray(traj["action"]).shape[0])
        return ne, nf

    n_tr_ep, n_tr_fr = _count(True)    # train[:95%]
    n_val_ep, n_val_fr = _count(False)  # train[95%:]
    ep = n_tr_ep + n_val_ep
    fr = n_tr_fr + n_val_fr
    logging.info(f"Repair episodes — train(95%)={n_tr_ep}  val(5%)={n_val_ep}  combined={ep}")
    logging.info(f"Repair windowed frames — train={n_tr_fr}  val={n_val_fr}  combined={fr}")
    return dict(tfds_total=total, n_train_ep=n_tr_ep, n_val_ep=n_val_ep,
                episodes=ep, frames=fr)


# =============================================================================
# Full run loop
# =============================================================================
def run_full(state, teacher_params, train_step, diag_step, preserve_iter, repair_iter,
             lr_fn, B_P, B_R, cov, coverage_ok):
    """10k-step combined two-stream run. No early-stop (fail-fast is the human reading wandb).
    Checkpoints every save_interval to a NEW dir; never V1 (red line 2). NaN grad -> stop."""
    num_steps = FLAGS.num_steps
    save_dir = os.path.abspath(FLAGS.save_dir)

    # red line 2: refuse to write anywhere near the read-only V1 weights
    assert save_dir != os.path.abspath(FLAGS.v1_path), "save_dir must not be the V1 dir"
    assert "checkpoints_distill_v2" not in save_dir, "refusing to write inside checkpoints_distill_v2 (V1, read-only)"

    epoch_steps = cov["frames"] / max(B_R, 1)
    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.name,
        config=dict(
            rho_R=FLAGS.rho_R, B_P=B_P, B_R=B_R,
            lr=LR_PEAK, lr_warmup=LR_WARMUP, lr_schedule="constant_flat_no_decay",
            weight_decay=WEIGHT_DECAY, clip_gradient=CLIP_GRADIENT, alpha=FLAGS.alpha,
            num_steps=num_steps, save_interval=FLAGS.save_interval, save_dir=save_dir,
            v1_path=FLAGS.v1_path, v1_step=FLAGS.v1_step, chunk_source="predicted", path="B",
            repair_episodes=cov["episodes"], repair_frames=cov["frames"],
            repair_steps_per_epoch=epoch_steps, coverage_ok=coverage_ok, seed=FLAGS.seed,
        ),
    )
    try:
        logging.info(f"wandb run URL: {wandb.run.url}")
    except Exception:
        logging.info("wandb run started (URL unavailable)")
    if B_R > 0:
        logging.info(f"repair epoch = {cov['frames']} frames / {B_R} per step = {epoch_steps:.1f} steps/epoch")
    else:
        logging.info("rho_R=0 CONTROL: repair stream OFF (B_R=0) — pure OXE continuation, no repair_action curve")
    logging.info(f"💾 save dir = {save_dir}  (every {FLAGS.save_interval} steps)")

    save_cb = SaveCallback(save_dir)
    diag_rng = jax.random.PRNGKey(0)
    has_repair = B_R > 0  # rho_R=0 control: repair stream off (everything else verbatim)

    for step in range(num_steps):
        pb = next(preserve_iter)
        if has_repair:
            rb = next(repair_iter)
            dropped = []
            combined = tree_concat(pb, rb, dropped)
        else:
            dropped = []
            combined = pb  # pure OXE; B_P already == batch_size, total still 128
        if step == 0:
            act_shape = np.asarray(combined["action"]).shape
            assert act_shape == (FLAGS.batch_size, 2, 4, 7), f"combined action {act_shape} != (128,2,4,7)"
            logging.info(f"✅ combined action shape = {act_shape}; has_repair={has_repair}; dropped non-shared keys = {dropped}")

        combined_dev = jax.device_put(combined)
        state, metrics, gnorm = train_step(state, teacher_params, combined_dev)

        lr = float(lr_fn(step))
        total = float(metrics["total"]); kd = float(metrics["kd"]); action = float(metrics["action"])
        gnf = float(gnorm)
        log = dict(total=total, kd=kd, action=action, grad_norm=gnf, lr=lr)

        # per-100 report-only sharded diagnostic: preserve always; repair only if the stream is on
        if step % 100 == 0:
            pre = tree_slice(combined_dev, slice(0, B_P))
            a_pre, k_pre = diag_step(state.model.params, teacher_params, pre, diag_rng)
            log.update(preserve_action=float(a_pre), preserve_KD=float(k_pre))
            if has_repair:
                rep = tree_slice(combined_dev, slice(B_P, FLAGS.batch_size))
                a_rep, k_rep = diag_step(state.model.params, teacher_params, rep, diag_rng)
                log.update(repair_action=float(a_rep), repair_KD=float(k_rep))

        wandb.log(log, step=step)

        if step < 10 or step % 50 == 0:
            extra = ""
            if "preserve_action" in log:
                extra = f" | preserve_action={log['preserve_action']:.4f}"
                if "repair_action" in log:
                    extra += f" repair_action={log['repair_action']:.4f}"
            logging.info(f"[full {step:05d}/{num_steps}] total={total:.4f} kd={kd:.4f} action={action:.4f} "
                         f"lr={lr:.2e} grad_norm={gnf:.4f}{extra}{'  NaN!' if not np.isfinite(gnf) else ''}")

        if not np.isfinite(gnf):
            logging.error(f"non-finite grad_norm at step {step} — stopping run.")
            wandb.finish(exit_code=1)
            raise SystemExit(1)

        # checkpoint every save_interval (label by step+1 -> first ckpt = save_interval)
        if (step + 1) % FLAGS.save_interval == 0:
            save_cb(state, step + 1)
            logging.info(f"💾 saved checkpoint @ step {step + 1} -> {save_dir}")

    if num_steps % FLAGS.save_interval != 0:
        save_cb(state, num_steps)
        logging.info(f"💾 final checkpoint @ step {num_steps}")
    wandb.finish()
    logging.info("🏁 full run complete")


# =============================================================================
# Main
# =============================================================================
def main(_):
    tf.config.set_visible_devices([], "GPU")  # TF only feeds data
    logging.set_verbosity(logging.INFO)

    # Mode select: exactly one of --smoke / --full must be set. Neither or both is rejected
    # so the full job can never start by accident (red lines 9/10).
    if FLAGS.smoke == FLAGS.full:
        logging.error("Choose exactly one mode: --smoke (10-step ingest gate) OR --full (real run). "
                      "Refusing to run with neither/both set.")
        raise SystemExit(2)
    mode = "smoke" if FLAGS.smoke else "full"

    B_R = round(FLAGS.rho_R * FLAGS.batch_size)
    B_P = FLAGS.batch_size - B_R
    logging.info(f"MODE={mode} | dose rho_R={FLAGS.rho_R} -> B_P={B_P}, B_R={B_R}, total={B_P + B_R}")

    # ---- 1. Models -------------------------------------------------------
    logging.info(f"Loading student (V1) from {FLAGS.v1_path} @ step {FLAGS.v1_step} (READ-ONLY)")
    student = OctoModel.load_pretrained(FLAGS.v1_path, step=FLAGS.v1_step)
    logging.info(f"Loading teacher (12L) from {FLAGS.teacher_checkpoint}")
    teacher = OctoModel.load_pretrained(FLAGS.teacher_checkpoint)

    # assert student is already 8L; we do NOT prune here
    trainer = OXEDistillationTrainer()
    trainer.student_model = student
    trainer.teacher_model = teacher
    n_student = trainer._count_layers(student.params)
    n_teacher = trainer._count_layers(teacher.params)
    assert n_student == 8, f"Expected 8L student, got {n_student}"
    assert n_teacher == 12, f"Expected 12L teacher, got {n_teacher}"
    logging.info(f"✅ student layers = {n_student}, teacher layers = {n_teacher}")

    text_processor = student.text_processor
    assert text_processor is not None, "student text_processor is None"

    # ---- 2. Loss (verbatim reuse: 0.5*KD + 0.5*action) -------------------
    loss_fn = trainer.make_loss_fn(alpha=FLAGS.alpha)
    teacher_params = teacher.params

    # ---- 3. Optimizer reset + fresh TrainState ---------------------------
    opt_kwargs = dict(
        learning_rate=dict(
            name="constant",          # linear warmup -> flat peak; no decay
            init_value=0.0,
            peak_value=LR_PEAK,
            warmup_steps=LR_WARMUP,
        ),
        weight_decay=WEIGHT_DECAY,
        clip_gradient=CLIP_GRADIENT,
    )
    tx, lr_fn, _param_norm_fn = create_optimizer(student.params, **opt_kwargs)
    state = TrainState.create(jax.random.PRNGKey(FLAGS.seed), student, tx)  # Adam moments = 0
    logging.info("✅ fresh TrainState (V1 weights only; optimizer state reset)")

    # ---- 4. JIT steps ----------------------------------------------------
    @jax.jit
    def train_step(state, teacher_params, combined_batch):
        rng, step_rng = jax.random.split(state.rng)
        (total, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, teacher_params, combined_batch, step_rng
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        gnorm = optax.global_norm(grads)
        return new_state, metrics, gnorm

    @jax.jit
    def diag_step(student_params, teacher_params, batch, rng):
        """Report-only: action + KD on a single-stream slice. NOT used for gradients."""
        _total, metrics = loss_fn(student_params, teacher_params, batch, rng)
        return metrics["action"], metrics["kd"]

    # ---- 5. Streams ------------------------------------------------------
    # B_R==0 (rho_R=0) control: repair stream OFF. Single-variable vs the repair run —
    # B_P already == batch_size (128), so the effective gradient batch size is unchanged;
    # we just do not build/iterate/score the eggplant stream. Everything else is verbatim.
    has_repair = B_R > 0
    logging.info("Building preserve (OXE) stream...")
    preserve_iter = build_preserve_iter(B_P, text_processor)
    if has_repair:
        logging.info("Building repair (path-B) stream...")
        repair_iter = build_repair_iter(B_R, text_processor)
        logging.info("Counting repair coverage (one-time: episodes + exact frames)...")
        cov = count_repair_coverage()
        coverage_ok = cov["episodes"] == 324
    else:
        logging.info("rho_R=0 CONTROL: repair stream OFF — eggplant loader not built, coverage not counted.")
        repair_iter = None
        cov = dict(tfds_total=None, n_train_ep=0, n_val_ep=0, episodes=0, frames=0)
        coverage_ok = True  # N/A for the no-repair control
    tfds_total, n_train, n_val = cov["tfds_total"], cov["n_train_ep"], cov["n_val_ep"]
    repair_frames = cov["frames"]

    if mode == "full":
        run_full(state, teacher_params, train_step, diag_step, preserve_iter, repair_iter,
                 lr_fn, B_P, B_R, cov, coverage_ok)
        return

    # ---- 6. Smoke loop ---------------------------------------------------
    n_steps = FLAGS.smoke_steps
    logging.info(f"🔥 SMOKE: {n_steps} steps, no save, no wandb")
    diag_rng = jax.random.PRNGKey(0)
    ratios = []
    last = {}
    for step in range(n_steps):
        pb = next(preserve_iter)
        if has_repair:
            rb = next(repair_iter)
            dropped = []
            combined = tree_concat(pb, rb, dropped)
        else:
            dropped = []
            combined = pb  # rho_R=0 control: pure OXE; B_P already == batch_size

        if step == 0:
            _print_struct("PRESERVE", pb)
            if has_repair:
                _print_struct("REPAIR", rb)
            _print_struct("COMBINED", combined)
            if dropped:
                logging.warning(f"Dropped non-shared keys during concat: {dropped}")
            act_shape = np.asarray(combined["action"]).shape
            assert act_shape == (FLAGS.batch_size, 2, 4, 7), \
                f"combined action shape {act_shape} != (128,2,4,7)"
            logging.info(f"✅ combined action shape = {act_shape}")

        combined_dev = jax.device_put(combined)
        state, metrics, gnorm = train_step(state, teacher_params, combined_dev)

        # report-only sharded diagnostic: preserve always; repair only if the stream is on
        pre_slice = tree_slice(combined_dev, slice(0, B_P))
        a_pre, k_pre = diag_step(state.model.params, teacher_params, pre_slice, diag_rng)
        a_pre = float(a_pre); k_pre = float(k_pre)
        if has_repair:
            rep_slice = tree_slice(combined_dev, slice(B_P, FLAGS.batch_size))
            a_rep, k_rep = diag_step(state.model.params, teacher_params, rep_slice, diag_rng)
            a_rep = float(a_rep); k_rep = float(k_rep)
            ratio = a_rep / max(a_pre, 1e-8)
            ratios.append(ratio)
        else:
            a_rep = k_rep = ratio = float("nan")  # no repair stream in the control

        gnf = float(gnorm)
        last = dict(
            total=float(metrics["total"]), kd=float(metrics["kd"]),
            action=float(metrics["action"]),
            a_pre=a_pre, a_rep=a_rep, k_pre=k_pre, k_rep=k_rep,
            ratio=ratio, gnorm=gnf,
        )
        rep_str = (f"repair_action={a_rep:.4f} ratio={ratio:.2f}x repair_KD={k_rep:.4f}"
                   if has_repair else "repair_stream=OFF (control)")
        logging.info(
            f"[smoke {step:02d}] total={last['total']:.4f} kd={last['kd']:.4f} "
            f"action={last['action']:.4f} | preserve_action={a_pre:.4f} preserve_KD={k_pre:.4f} | "
            f"{rep_str} | grad_norm={gnf:.4f} {'NaN!' if not np.isfinite(gnf) else ''}"
        )

    # ---- 7. Hard gate verdict (red line 6) -------------------------------
    grad_finite = np.isfinite(last.get("gnorm", np.nan))
    logging.info("=" * 70)
    if has_repair:
        med_ratio = float(np.median(ratios))
        logging.info(f"REPAIR EPISODE COVERAGE: train={n_train} + val={n_val} = {n_train + n_val} "
                     f"(tfds_total={tfds_total})  -> {'OK (324)' if coverage_ok else 'NOT 324 — investigate'}")
        logging.info(f"INGEST-SCALE GATE: median repair/preserve action ratio = {med_ratio:.2f}x")
        logging.info(f"  ideal ~0.9-1.1, acceptable <~3x, FAIL if >=10x")
        ratio_ok = med_ratio < 10.0
    else:
        logging.info("rho_R=0 CONTROL: repair stream OFF — coverage N/A, ingest-ratio gate N/A.")
        logging.info("  Gate reduces to: pure-OXE combined batch is (128,2,4,7) + grads finite/no-NaN.")
        ratio_ok = True
    logging.info(f"GRAD NORM finite/no-NaN: {grad_finite}")
    gate_pass = ratio_ok and grad_finite and coverage_ok
    logging.info(f"SMOKE HARD GATE: {'PASS' if gate_pass else 'FAIL — STOP and report'}")
    logging.info("Note: this gate only checks ingest scale/format, NOT repair quality or success "
                 "(red line 7: only local closed-loop counts).")
    logging.info("=" * 70)
    raise SystemExit(0 if gate_pass else 1)


if __name__ == "__main__":
    app.run(main)
