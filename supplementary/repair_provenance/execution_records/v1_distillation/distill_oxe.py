import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
from functools import partial
from typing import Dict

from absl import app, flags, logging
from flax.core import unfreeze, freeze
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model import OctoModel
from octo.model.octo_module import OctoModule
from octo.utils import jax_utils
from octo.utils.train_callbacks import SaveCallback
from octo.utils.train_utils import (
    create_optimizer,
    process_text,
    Timer,
    TrainState,
)

FLAGS = flags.FLAGS

# =============================================================================
# 1. 实验配置
# =============================================================================
flags.DEFINE_string("name", "oxe_distill_12to8", "Experiment name (用于wandb run名称)")
flags.DEFINE_bool("debug", False, "Debug模式: 只跑10步, 不存档, 不上wandb")
flags.DEFINE_string("teacher_checkpoint", None, "Teacher checkpoint路径", required=True)
flags.DEFINE_string("save_dir", "checkpoints_distill_v2", "Checkpoint保存目录")

# 数据目录 = 你的实际路径 downloads
flags.DEFINE_string("data_dir", "downloads", "OXE数据根目录")
flags.DEFINE_string("data_mix", "oxe_magic_soup", "OXE mix名称")

flags.DEFINE_integer("batch_size", 128, "Batch size (A100 80G建议128)")
flags.DEFINE_integer("student_layers", 8, "Student目标层数")
flags.DEFINE_float("alpha", 0.5, "KD loss权重, action loss权重 = 1 - alpha")
flags.DEFINE_integer("num_steps", 300000, "总训练步数")

# wandb配置
flags.DEFINE_string("wandb_project", "octo_oxe_distill", "wandb项目名")
flags.DEFINE_string("wandb_entity", None, "wandb entity (team名, 没有就留None)")

config_flags.DEFINE_config_dict(
    "config",
    ConfigDict({
        "seed": 42,
        "optimizer": {
            "learning_rate": {
                "name": "cosine",
                "init_value": 0.0,
                "peak_value": 3e-5,  # 降低LR防止LR Shock破坏继承的特征空间
                "warmup_steps": 2000,
                "decay_steps": 300000,  # main()里会覆盖成实际num_steps
                "end_value": 1e-6,
            },
            "weight_decay": 0.01,
            "clip_gradient": 1.0,
        },
        "log_interval": 100,
        "save_interval": 5000,  # 每5000步保存一次checkpoint
    })
)

# =============================================================================
# 2. 蒸馏核心
# =============================================================================
class OXEDistillationTrainer:
    """
    全量OXE蒸馏：Teacher(12层 Octo-Base) → Student(8层)

    层路径（基于源码 transformer.py line 223）：
      params
        └── octo_transformer
              └── BlockTransformer_0
                    └── Transformer_0
                          ├── encoderblock_0
                          ├── ...
                          └── encoderblock_11

    Expert Surgery索引（12→8）：[0, 1, 3, 4, 6, 8, 10, 11]
      - 保留 0  : 接收tokenizer输出的首层
      - 保留 11 : 直连 DiffusionActionHead 的末层
      - 中间均匀采样填充剩余6层
    """

    _TRANSFORMER_PATH = ['octo_transformer', 'BlockTransformer_0', 'Transformer_0']
    _LAYER_PREFIX     = 'encoderblock_'
    _EXPERT_INDICES   = [0, 1, 3, 4, 6, 8, 10, 11]

    def __init__(self):
        self.teacher_model: OctoModel = None
        self.student_model: OctoModel = None

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _get_transformer_block(self, params: Dict) -> Dict:
        """沿固定路径取出包含 encoderblock_N 的字典（原地引用）"""
        curr = params
        for k in self._TRANSFORMER_PATH:
            curr = curr[k]
        return curr

    def _count_layers(self, params: Dict) -> int:
        block = self._get_transformer_block(params)
        n = 0
        while f'{self._LAYER_PREFIX}{n}' in block:
            n += 1
        return n

    # ------------------------------------------------------------------
    # Teacher加载
    # ------------------------------------------------------------------
    def load_teacher(self, checkpoint_path: str) -> OctoModel:
        logging.info(f"Loading Teacher: {checkpoint_path}")
        self.teacher_model = OctoModel.load_pretrained(checkpoint_path)
        n = self._count_layers(self.teacher_model.params)
        assert n == 12, f"Expected 12-layer Teacher, got {n}"
        logging.info(f"✅ Teacher loaded: {n} layers confirmed")
        return self.teacher_model

    # ------------------------------------------------------------------
    # Student创建
    # ------------------------------------------------------------------
    def create_student(self, target_layers: int = 8) -> OctoModel:
        assert target_layers == 8, "本脚本专为 12→8 设计"

        # Step 1: 深拷贝 + unfreeze
        # FIX: Flax params 是 FrozenDict，不支持原地修改
        # 必须 unfreeze 解冻为普通 dict，修改后 freeze 重新冻结
        student_params = unfreeze(copy.deepcopy(self.teacher_model.params))
        student_config = copy.deepcopy(self.teacher_model.config)

        # Step 2: Expert Surgery 剪枝 params
        block = self._get_transformer_block(student_params)  # 原地引用

        new_layers = {
            f'{self._LAYER_PREFIX}{new_i}': block[f'{self._LAYER_PREFIX}{old_i}']
            for new_i, old_i in enumerate(self._EXPERT_INDICES)
        }
        # 遍历 list 副本再删除，防止运行时字典大小改变报错
        for k in list(block.keys()):
            if k.startswith(self._LAYER_PREFIX):
                del block[k]
        block.update(new_layers)

        # 重新冻结，符合 Flax 输入规范
        student_params = freeze(student_params)

        # 验证
        actual = self._count_layers(student_params)
        assert actual == target_layers, f"Pruning failed: got {actual}"
        logging.info(f"✅ Params pruned: 12→{actual} | indices={self._EXPERT_INDICES}")

        # Step 3: 更新 config 中的 num_layers
        # 路径依据: octo_model.py line 317 → OctoModule.create(**config["model"])
        # OctoModule.create 接收 transformer_kwargs dict，其中含 num_layers
        student_config['model']['transformer_kwargs']['num_layers'] = target_layers
        confirmed = student_config['model']['transformer_kwargs']['num_layers']
        assert confirmed == target_layers, "Config update failed"
        logging.info(f"✅ Config: transformer_kwargs.num_layers = {confirmed}")

        # Step 4: 重建计算图，绑定剪枝后的 params
        new_module = OctoModule.create(**student_config['model'])
        self.student_model = self.teacher_model.replace(
            params=student_params,
            config=student_config,
            module=new_module,
        )
        logging.info("✅ Student model ready")
        return self.student_model

    # ------------------------------------------------------------------
    # Loss函数
    # ------------------------------------------------------------------
    def make_loss_fn(self, alpha: float):
        """
        Loss = alpha * KD_loss + (1-alpha) * action_loss

        设计要点：
        1. FIX: RNG 拆成三份，transformer/head/step 互不污染
           原因: DiffusionActionHead 内部用 make_rng("dropout") 做加噪采样，
                 与 Transformer dropout 共用同一 rng 会产生强相关性，破坏多样性

        2. FIX: KD loss 覆盖全部 token group（obs + task + readout_action）
           原因: Action Head 唯一依赖的是 readout_action token，
                 只对 obs 做 KD 等于完全没有对 action readout 进行知识蒸馏，
                 Student 的动作预测能力无法从 Teacher 获得引导

        3. 使用 TokenGroup.mask 而非手动广播 pad_mask
           原因: TokenGroup.mask 是 Octo 内部的精确 token 级 mask，
                 覆盖了模态 dropout 等复杂情况，比 timestep_pad_mask 广播更准确

        4. teacher_params 不套 stop_gradient（整体参数字典级别）
           原因: value_and_grad(argnums=0) 只对第0个参数(student_params)求导，
                 teacher_params 本身就不产生梯度，无需包裹
                 stop_gradient 只在 token 输出层面截断（更精准的语义）
        """
        student_module = self.student_model.module
        teacher_module = self.teacher_model.module

        def loss_fn(student_params, teacher_params, batch, rng):
            obs             = batch["observation"]
            tasks           = batch["task"]
            pad_mask        = obs["timestep_pad_mask"]   # (B, T)
            actions         = batch["action"]
            action_pad_mask = batch["action_pad_mask"]

            # FIX: 拆成三份独立 RNG，避免计算流污染
            rng, dropout_rng_trans, dropout_rng_head = jax.random.split(rng, 3)

            # ── Student Step1: 只跑 transformer ──
            # method="octo_transformer" → OctoTransformer.__call__
            # 返回 Dict[str, TokenGroup]，包含 obs / task / readout_action
            student_transformer_out = student_module.apply(
                {"params": student_params},
                obs, tasks, pad_mask,
                train=True,
                rngs={"dropout": dropout_rng_trans},
                method="octo_transformer",
            )

            # ── Student Step2: action head loss ──
            # DiffusionActionHead.loss 签名:
            #   (transformer_outputs, actions, timestep_pad_mask, action_pad_mask, train)
            # 内部用 self.make_rng("dropout") 获取 rng，必须在 apply 内调用
            action_loss, action_metrics = student_module.apply(
                {"params": student_params},
                student_transformer_out,
                actions,
                pad_mask,
                action_pad_mask,
                train=True,
                rngs={"dropout": dropout_rng_head},
                method=lambda m, *a, **kw: m.heads["action"].loss(*a, **kw),
            )

            # ── Teacher: 独立前向，不进梯度图 ──
            teacher_transformer_out = teacher_module.apply(
                {"params": teacher_params},
                obs, tasks, pad_mask,
                train=False,
                method="octo_transformer",
            )

            # ── FIX: 全量 Masked Feature Matching ──
            # 遍历所有 token group: obs, task, readout_action
            # 每个 group 用自己的 TokenGroup.mask（比 pad_mask 广播更精准）
            kd_loss = jnp.zeros(())
            kd_metrics = {}
            valid_groups = 0

            for key in student_transformer_out.keys():
                if key not in teacher_transformer_out:
                    continue

                s_tg = student_transformer_out[key]
                t_tg = teacher_transformer_out[key]

                s_tok = s_tg.tokens.astype(jnp.float32)
                t_tok = jax.lax.stop_gradient(t_tg.tokens.astype(jnp.float32))

                # Cosine Distance KD Loss
                # 公式: 1 - cosine_similarity(s, t)，范围[0, 2]
                # 优点1: 量级天然~1.0，与action loss(1.0~1.5)对齐，解决scale mismatch
                # 优点2: 只关注特征方向，不受幅度影响，更适合蒸馏场景
                # 优点3: 无硬编码系数，自适应
                # 数值验证: 随机向量→~1.0, 完全对齐→0.0, 完全相反→2.0
                s_norm = s_tok / (jnp.linalg.norm(s_tok, axis=-1, keepdims=True) + 1e-6)
                t_norm = t_tok / (jnp.linalg.norm(t_tok, axis=-1, keepdims=True) + 1e-6)

                # cos_sim shape: (..., N_tok)，对特征维度D求点积
                cos_sim = jnp.sum(s_norm * t_norm, axis=-1)
                cos_dist = 1.0 - cos_sim  # (B, T, N_tok) 或 (B, N_tok)

                # TokenGroup.mask shape:
                # - prefix group (task): (B, N_tok)
                # - timestep group (obs, readout): (B, T, N_tok)
                mask = s_tg.mask.astype(jnp.float32)  # 无需扩展，cos_dist已无特征维

                group_loss = jnp.sum(cos_dist * mask) / (jnp.sum(mask) + 1e-6)

                kd_loss = kd_loss + group_loss
                kd_metrics[f"kd/{key}"] = group_loss
                valid_groups += 1

            # 各 group 取平均，防止 group 数量不同时 loss 尺度漂移
            kd_loss = kd_loss / jnp.maximum(valid_groups, 1)

            total = alpha * kd_loss + (1.0 - alpha) * action_loss

            metrics = {
                "total":  total,
                "kd":     kd_loss,
                "action": action_loss,
                **kd_metrics,
                **{f"action/{k}": v for k, v in action_metrics.items()},
            }
            return total, metrics

        return loss_fn


# =============================================================================
# 3. 主程序
# =============================================================================
def main(_):
    jax_utils.initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")  # TF只负责数据，不抢显存

    num_steps = 10 if FLAGS.debug else FLAGS.num_steps

    # ── wandb初始化 ──
    if jax.process_index() == 0 and not FLAGS.debug:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=FLAGS.name,
            config={
                **FLAGS.config.to_dict(),
                "num_steps":     num_steps,
                "batch_size":    FLAGS.batch_size,
                "alpha":         FLAGS.alpha,
                "student_layers": FLAGS.student_layers,
                "data_mix":      FLAGS.data_mix,
                "data_dir":      FLAGS.data_dir,
            },
            # 实时监控：每100步同步一次
            settings=wandb.Settings(x_disable_stats=False),
        )
        logging.info(f"✅ wandb initialized: {FLAGS.wandb_project}/{FLAGS.name}")

    mesh        = Mesh(jax.devices(), axis_names="batch")
    replicated  = NamedSharding(mesh, PartitionSpec())
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    # ── 1. 模型 ──────────────────────────────────────────────────────────
    trainer = OXEDistillationTrainer()
    trainer.load_teacher(FLAGS.teacher_checkpoint)

    # 从 Teacher config 同步 window_size（防止 shape mismatch）
    try:
        window_size = trainer.teacher_model.config[
            'dataset_kwargs']['traj_transform_kwargs']['window_size']
    except (KeyError, TypeError):
        window_size = 2
        logging.warning(f"window_size not in Teacher config, defaulting to {window_size}")
    logging.info(f"✅ window_size = {window_size}")

    text_processor = trainer.teacher_model.text_processor
    assert text_processor is not None, "Teacher text_processor is None"

    trainer.create_student(FLAGS.student_layers)

    # ── 2. 数据集 ─────────────────────────────────────────────────────────
    # 注意：你的数据目录是 downloads
    # 其中 bridge 文件夹对应 oxe 中的 bridge_dataset
    # make_oxe_dataset_kwargs_and_weights 会自动处理每个数据集的 standardize_fn
    # 和 image_obs_keys，不需要手动指定
    logging.info(f"Building OXE dataset: {FLAGS.data_mix} from {FLAGS.data_dir}")

    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        FLAGS.data_mix,
        data_dir=FLAGS.data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,     # 各数据集 proprio 空间维度不统一，关闭
        load_language=True,
        action_proprio_normalization_type="normal",
    )
    logging.info(f"✅ {len(dataset_kwargs_list)} datasets loaded from mix")

    interleaved_kwargs = dict(
        dataset_kwargs_list=dataset_kwargs_list,
        sample_weights=sample_weights,
        train=True,
        # 从 200000 降到 50000，防止 OOM
        # 你的机器 251GB RAM 理论够用，但 OXE 多视角图像解码后占用难以预估
        # 跑起来后观察 htop，如果 RAM 余量大可以适当调高
        shuffle_buffer_size=50000,
        traj_transform_kwargs=dict(
            window_size=window_size,
            action_horizon=4,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        batch_size=FLAGS.batch_size,
        balance_weights=True,   # 全量 mix 必须 balance，防止大数据集淹没小数据集
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    def make_data_iter():
        ds = make_interleaved_dataset(**interleaved_kwargs)
        def _process(batch):
            batch = process_text(batch, text_processor)
            batch.pop("dataset_name", None)
            return batch
        def _shard(batch):
            return multihost_utils.host_local_array_to_global_array(
                batch, mesh, PartitionSpec("batch")
            )
        return map(_shard, map(_process, ds.iterator()))

    train_iter = make_data_iter()

    # 验证数据 pipeline
    logging.info("Verifying data pipeline...")
    dummy = next(train_iter)
    logging.info(
        f"✅ Data OK | "
        f"action shape: {dummy['action'].shape} | "
        f"obs keys: {list(dummy['observation'].keys())}"
    )
    del dummy

    # ── 3. 优化器 & TrainState ───────────────────────────────────────────
    # debug模式下num_steps=10，比warmup_steps=2000小会导致optax报负数decay_steps
    # 所以debug时warmup_steps也缩小为1，保证 decay_steps > warmup_steps
    warmup_steps = 1 if FLAGS.debug else FLAGS.config["optimizer"]["learning_rate"]["warmup_steps"]
    opt_cfg = {
        "learning_rate": {
            "name": "cosine",
            "init_value":    FLAGS.config["optimizer"]["learning_rate"]["init_value"],
            "peak_value":    FLAGS.config["optimizer"]["learning_rate"]["peak_value"],
            "warmup_steps":  warmup_steps,
            "decay_steps":   num_steps,  # debug=10, 正式=300000
            "end_value":     FLAGS.config["optimizer"]["learning_rate"]["end_value"],
        },
        "weight_decay":  FLAGS.config["optimizer"]["weight_decay"],
        "clip_gradient": FLAGS.config["optimizer"]["clip_gradient"],
    }

    tx, lr_fn, grad_norm_fn = create_optimizer(trainer.student_model.params, **opt_cfg)
    state = TrainState.create(
        jax.random.PRNGKey(FLAGS.config["seed"]),
        trainer.student_model,
        tx,
    )

    if not FLAGS.debug:
        save_cb = SaveCallback(FLAGS.save_dir)
        logging.info(f"✅ Checkpoints will be saved to {FLAGS.save_dir} every {FLAGS.config['save_interval']} steps")

    # ── 4. JIT 训练步 ────────────────────────────────────────────────────
    loss_fn = trainer.make_loss_fn(alpha=FLAGS.alpha)

    # Teacher params 独立传入，不放进 state，不被 donate_argnums 销毁
    teacher_params = trainer.teacher_model.params

    @partial(
        jax.jit,
        in_shardings=(replicated, replicated, dp_sharding),
        out_shardings=(replicated, replicated),
        donate_argnums=0,   # 只 donate student state，节省显存
    )
    def train_step(state, teacher_params, batch):
        rng, step_rng = jax.random.split(state.rng)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params,
            teacher_params,
            batch,
            step_rng,
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        info["lr"]        = lr_fn(state.step)
        info["grad_norm"] = grad_norm_fn(grads)
        return new_state, info

    # ── 5. 训练循环 ──────────────────────────────────────────────────────
    logging.info(
        f"🚀 Starting distillation | "
        f"steps={num_steps} | alpha={FLAGS.alpha} | "
        f"batch={FLAGS.batch_size} | save_every={FLAGS.config['save_interval']}"
    )
    timer = Timer()
    log_interval  = FLAGS.config["log_interval"]
    save_interval = FLAGS.config["save_interval"]

    for i in tqdm.tqdm(range(num_steps), desc="Distilling"):
        timer.tick("total")
        try:
            batch = next(train_iter)
        except StopIteration:
            logging.warning("⚠️ Iterator exhausted — rebuilding data pipeline")
            train_iter = make_data_iter()
            batch = next(train_iter)

        state, info = train_step(state, teacher_params, batch)
        timer.tock("total")

        # 每 log_interval 步上报 wandb + 本地日志
        if (i + 1) % log_interval == 0:
            info = jax.device_get(info)
            if jax.process_index() == 0:
                sps = log_interval / timer.get_average_times(reset=True).get("total", 1.0)
                eta_h = (num_steps - i - 1) / sps / 3600

                logging.info(
                    f"Step {i+1:6d}/{num_steps} | "
                    f"total={info['total']:.4f} | "
                    f"kd={info['kd']:.4f} | "
                    f"action={info['action']:.4f} | "
                    f"lr={info['lr']:.2e} | "
                    f"grad={info['grad_norm']:.3f} | "
                    f"sps={sps:.1f} | "
                    f"ETA={eta_h:.1f}h"
                )
                if not FLAGS.debug:
                    # flatten_dict 把嵌套 dict 展平成 wandb 可识别的格式
                    # 例如 kd/obs, kd/readout_action, action/mse 等
                    wandb.log(flatten_dict(info, sep="/"), step=i + 1)

        # 每 save_interval 步保存 checkpoint
        # 所有进程必须同步进入 save_cb，Orbax 内部协调，只让 rank 0 落盘
        if not FLAGS.debug and (i + 1) % save_interval == 0:
            save_cb(state, i + 1)
            if jax.process_index() == 0:
                logging.info(f"💾 Checkpoint saved @ step {i+1} → {FLAGS.save_dir}")

    # 最终 checkpoint
    if not FLAGS.debug:
        save_cb(state, num_steps)
        if jax.process_index() == 0:
            logging.info(f"💾 Final checkpoint saved @ step {num_steps}")
            wandb.finish()

    logging.info("🏆 Distillation complete!")


if __name__ == "__main__":
    app.run(main)