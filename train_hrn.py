import os
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import clip
from PIL import Image
import torch.nn.functional as F
from model_hrn_clip import HRNCLIP_Progressive, HRNCLIP_ViT
from dataset_multilang import MultilangDataset
from confidence_calibrator import HierarchicalConfidenceCalibrator
from utils_multilang import (
    get_hierarchy_targets,
    build_hierarchy_matrices,
    cosine_anneal_schedule,
    calculate_hierarchical_f1,
    calculate_severity_of_mistake,
    calculate_accuracy_metrics,
)


def arg_parse():
    parser = argparse.ArgumentParser(description="HRN-CLIP Progressive with Hierarchical Confidence Calibration")
    parser.add_argument("--worker", default=4, type=int, help="DataLoader workers")
    parser.add_argument("--model", type=str, default=None, help="Path of trained checkpoint to resume")
    parser.add_argument('--backbone', type=str, default='RN50',
                        choices=['RN50', 'RN50x64', 'RN50x4', 'RN50x16', 'ViT-B-16', 'ViT-B-32', 'ViT-L-14'],
                        help='CLIP backbone name')
    parser.add_argument("--clip_pretrained", type=str, default=None,
                        help="CLIP visual pretrained path (.pt/.pth)")
    parser.add_argument("--freeze_clip", action="store_true",
                        help="Freeze CLIP visual encoder, only train HRN heads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr_adjt", type=str, default="Cos", choices=["Cos", "Step"])
    parser.add_argument("--device", nargs="+", default=["0"])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results")

    # Loss options
    parser.add_argument("--use_adaptive_ce", action="store_true",
                        help="Use HCAL-style adaptive weighting for Macro/Script/Language CE losses")
    parser.add_argument("--adaptive_gamma", type=float, default=0.5,
                        help="Temperature for adaptive CE weighting")
    parser.add_argument("--use_tree", action="store_true",
                        help="Use TreeLoss as an auxiliary hierarchy regularizer")
    parser.add_argument("--tree_weight", type=float, default=0.1,
                        help="Weight of TreeLoss when --use_tree is enabled")

    # Confidence calibration options
    parser.add_argument("--use_conf_calib", action="store_true",
                        help="Use prototype-based hierarchical confidence calibration during evaluation")
    parser.add_argument("--conf_beta", type=float, default=10.0,
                        help="Distance-to-confidence sharpness for confidence calibration")
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                        help="If >0, report metrics on confident samples whose calibrated confidence >= threshold")
    parser.add_argument("--calib_start_epoch", type=int, default=0,
                        help="Start fitting confidence calibrator from this epoch index")
    parser.add_argument("--eval_only",action="store_true",
                        help="Only evaluate checkpoint without training"
)

    return parser.parse_args()


class Logger:
    def __init__(self, save_dir, save_name):
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{save_name}.json")
        self.history = []
        self.best_epoch = 0
        self.best_h_f1 = 0.0

    def log_and_print(self, epoch, train_m, test_m, time_cost, lr=None):
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%m-%d %H:%M"),
            "lr": lr,
            "train": train_m,
            "test": test_m,
            "time": round(time_cost, 1),
        }
        self.history.append(entry)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump({
                "history": self.history,
                "best": {"epoch": self.best_epoch, "h_f1": self.best_h_f1}
            }, f, indent=2)

        is_best = "★" if test_m["hierarchical_f1"] >= self.best_h_f1 else " "
        extra = ""
        if "confident_h_f1" in test_m:
            extra = f" | Conf-HF1:{test_m['confident_h_f1']:.3f} Unrel:{test_m['unreliable_rate']:.1f}%"

        print(
            f"[{is_best}]Ep{epoch:02d} | "
            f"Train Acc:{train_m['accuracy']:.1f}% H-F1:{train_m['hierarchical_f1']:.3f} | "
            f"Test Acc:{test_m['accuracy']:.1f}% H-F1:{test_m['hierarchical_f1']:.3f} "
            f"Sev:{test_m['severity_of_mistake']:.3f}{extra} | {time_cost:.1f}s"
        )

        return test_m["hierarchical_f1"] > self.best_h_f1

    def update_best(self, epoch, h_f1):
        self.best_epoch = epoch
        self.best_h_f1 = h_f1

    def print_final_summary(self):
        if not self.history:
            return
        best = self.history[self.best_epoch]
        m = best["test"]

        print("\n" + "=" * 80)
        print(f"Training Summary | Best Epoch: {self.best_epoch} | H-F1: {self.best_h_f1:.4f}")
        print("-" * 80)
        print(f"Level 1 (Macro)     Acc: {m['macro_acc']:>6.2f}%")
        print(f"Level 2 (Script)    Acc: {m['script_acc']:>6.2f}%")
        print(f"Level 3 (Language)  Acc: {m['language_acc']:>6.2f}%")
        print("-" * 80)
        print(f"Overall Accuracy:          {m['accuracy']:>6.2f}%")
        print(f"Hierarchical F1 (H-F1):    {m['hierarchical_f1']:>6.4f}")
        print(f"Severity of Mistake:       {m['severity_of_mistake']:>6.4f}")
        if "confident_h_f1" in m:
            print("-" * 80)
            print(f"Confident H-F1:            {m['confident_h_f1']:>6.4f}")
            print(f"Confident Accuracy:        {m['confident_accuracy']:>6.2f}%")
            print(f"Unreliable Rate:           {m['unreliable_rate']:>6.2f}%")
        print("=" * 80)
        print(f"Log: {self.log_file}")


def unwrap_model(net):
    return net.module if hasattr(net, "module") else net


def save_checkpoint(net, optimizer, epoch, save_dir, save_name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"best_{save_name}.pth")
    torch.save({
        "epoch": epoch,
        "state_dict": unwrap_model(net).state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    

def infer_clip_visual_cfg(state_dict):
    keys = list(state_dict.keys())
    has_visual_prefix = any(k.startswith('visual.') for k in keys)

    def vkey(name):
        return f"visual.{name}" if has_visual_prefix else name

    cfg = {}

    if vkey('proj') in state_dict:
        cfg['type'] = 'vit'

        # width = conv1.out_channels
        cfg['width'] = state_dict[vkey('conv1.weight')].shape[0]

        # layers = Transformer block num
        cfg['layers'] = len([k for k in keys if k.endswith('.attn.in_proj_weight')])

        # patch_size = conv1.kernel_size
        cfg['patch_size'] = state_dict[vkey('conv1.weight')].shape[-1]

        # input_resolution = patch_size * grid_size
        grid_size = round((state_dict[vkey('positional_embedding')].shape[0] - 1) ** 0.5)
        cfg['input_resolution'] = cfg['patch_size'] * grid_size

        # heads = width // 64
        cfg['heads'] = cfg['width'] // 64

        # output_dim = proj.shape[1]  (proj: [width, output_dim])
        cfg['output_dim'] = state_dict[vkey('proj')].shape[1]

    else:
        cfg['type'] = 'resnet'

        if has_visual_prefix:
            counts = [len(set(k.split(".")[2] for k in keys if k.startswith(f"visual.layer{b}")))
                      for b in [1, 2, 3, 4]]
        else:
            counts = [len(set(k.split(".")[1] for k in keys if k.startswith(f"layer{b}")))
                      for b in [1, 2, 3, 4]]
        cfg['layers'] = tuple(counts)

        # width = layer1.0.conv1.out_channels
        cfg['width'] = state_dict[vkey('layer1.0.conv1.weight')].shape[0]

        # input_resolution = 32 * output_width
        output_width = round((state_dict[vkey('attnpool.positional_embedding')].shape[0] - 1) ** 0.5)
        cfg['input_resolution'] = output_width * 32

        # heads = width * 32 // 64
        cfg['heads'] = cfg['width'] * 32 // 64
        cfg['patch_size'] = None

        if 'text_projection' in state_dict:
            cfg['output_dim'] = state_dict['text_projection'].shape[1]
        elif vkey('attnpool.c_proj.weight') in state_dict:
            cfg['output_dim'] = state_dict[vkey('attnpool.c_proj.weight')].shape[1]
        else:
            cfg['output_dim'] = 512  # fallback

    return cfg



def build_net(args, dataset_info, device):
    if args.clip_pretrained is None:
        args.clip_pretrained = os.path.join('pretrained', f'{args.backbone}.pt')

    state = None
    if args.clip_pretrained and os.path.exists(args.clip_pretrained):
        print(f"[Net] Inspecting pretrained: {args.clip_pretrained}")
        try:
            model, _ = clip.load(args.clip_pretrained, device=device)
            state = model.state_dict()
            print("[Net] Loaded via clip.load() for inspection")
        except Exception as e:
            print(f"[Net] clip.load() failed ({e}), using torch.load()")
            state = torch.load(args.clip_pretrained, map_location=device)
            if isinstance(state, torch.jit.ScriptModule):
                state = state.state_dict()
            elif hasattr(state, 'state_dict') and callable(getattr(state, 'state_dict')):
                state = state.state_dict()

    cfg = infer_clip_visual_cfg(state) if state else {
        'type': 'resnet', 'layers': (3, 4, 6, 3), 'width': 64,
        'heads': 32, 'output_dim': 512, 'input_resolution': 224, 'patch_size': None
    }
    print(f"[Net] Auto-inferred cfg: {cfg}")
    if cfg['type'] == 'resnet':
        if args.backbone == 'RN50x64':
            cfg['input_resolution'] = 448
        elif args.backbone == 'RN50x4':
            cfg['input_resolution'] = 288
        elif args.backbone == 'RN50x16':
            cfg['input_resolution'] = 384
        else:
            cfg['input_resolution'] = 224
    else:  # vit
        if '336' in args.backbone:
            cfg['input_resolution'] = 336
        else:
            cfg['input_resolution'] = 224

    if cfg['type'] == 'resnet':
        net = HRNCLIP_Progressive(
            layers=cfg['layers'],
            output_dim=cfg['output_dim'],
            heads=cfg['heads'],
            num_macros=dataset_info['num_macros'],
            num_scripts=dataset_info['num_scripts'],
            num_languages=dataset_info['num_languages'],
            input_resolution=cfg['input_resolution'],
            width=cfg['width'],
            feature_size=512
        ).to(device)
    else:
        net = HRNCLIP_ViT(
            num_macros=dataset_info['num_macros'],
            num_scripts=dataset_info['num_scripts'],
            num_languages=dataset_info['num_languages'],
            input_resolution=cfg['input_resolution'],
            patch_size=cfg['patch_size'],
            width=cfg['width'],
            layers=cfg['layers'],
            heads=cfg['heads'],
            output_dim=cfg['output_dim'],
            feature_size=512
        ).to(device)

    if state is not None:
        has_visual_prefix = any(k.startswith('visual.') for k in state.keys())

        if cfg['type'] == 'resnet':
            if has_visual_prefix:
                state = {k.replace('visual.', ''): v for k, v in state.items()
                         if k.startswith('visual.')}
                print("[Net] Extracted visual branch for ResNet")
            model_dict = net.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered)
            net.load_state_dict(model_dict, strict=False)
            print(f"[Net] Loaded {len(filtered)} / {len(model_dict)} layers into ResNet")
        else:
            if has_visual_prefix:
                vit_state = {k.replace('visual.', ''): v for k, v in state.items()
                             if k.startswith('visual.')}
                print("[Net] Extracted visual branch for ViT")
            else:
                vit_state = state
            model_dict = net.visual.state_dict()
            filtered = {k: v for k, v in vit_state.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered)
            net.visual.load_state_dict(model_dict, strict=False)
            print(f"[Net] Loaded {len(filtered)} / {len(model_dict)} layers into ViT.visual")

    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        net.load_state_dict(state_dict, strict=False)
        print(f"[Net] Resumed from {args.model}")

    if args.freeze_clip:
        hrn_keywords = [
            'macro_conv', 'script_conv', 'language_conv',
            'macro_proj', 'script_proj', 'language_proj',
            'classifier_macro', 'classifier_script',
            'classifier_language', 'classifier_language_softmax'
        ]
        if cfg['type'] == 'vit':
            hrn_keywords.append('attnpool')

        frozen_names = []
        for name, param in net.named_parameters():
            if any(k in name for k in hrn_keywords):
                continue
            param.requires_grad = False
            frozen_names.append(name)
        print(f"[Net] Frozen {len(frozen_names)} params. Only HRN heads are trainable.")

    return net


class ResizeWithPadding:
    def __init__(self, target_size=224, fill_color=(0, 0, 0), interpolation=Image.BICUBIC):
        self.target_size = target_size
        self.fill_color = fill_color
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        ratio = self.target_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized_img = img.resize((new_w, new_h), self.interpolation)

        padded_img = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        padded_img.paste(
            resized_img,
            ((self.target_size - new_w) // 2, (self.target_size - new_h) // 2)
        )
        return padded_img


def adaptive_weighted_loss(losses, gamma=0.5):
    with torch.no_grad():
        loss_values = torch.stack([l.detach() for l in losses])
        weights = torch.softmax(loss_values / gamma, dim=0)

    total = sum(w * l for w, l in zip(weights, losses))
    return total, weights


def compute_training_loss(
    logits_m,
    logits_s,
    logits_l,
    targets,
    CELoss,
    device,
    dataset_info,
    use_adaptive_ce=False,
    adaptive_gamma=0.5,
    tree_loss_fn=None,
    use_tree=False,
    tree_weight=0.1,
    log_vars_hcd=None,
):
    num_m = dataset_info["num_macros"]
    num_s = dataset_info["num_scripts"]
    num_l = dataset_info["num_languages"]
    lang_offset = num_m + num_s

    macro_t, script_t, target_sig = get_hierarchy_targets(targets, device, dataset_info)
    language_t = targets - lang_offset

    loss_m = CELoss(logits_m, macro_t)
    loss_s = CELoss(logits_s, script_t)
    loss_l = CELoss(logits_l, language_t)

    if use_adaptive_ce:
        loss_ce, ce_weights = adaptive_weighted_loss(
            [loss_m, loss_s, loss_l],
            gamma=adaptive_gamma
        )
    else:
        loss_ce = loss_m + loss_s + loss_l
        ce_weights = torch.tensor([1.0, 1.0, 1.0], device=device)

    with torch.no_grad():
        lang2script_mat, script2macro_mat = build_hierarchy_matrices(dataset_info, device)
        script2macro_idx = script2macro_mat.argmax(dim=1)     
        lang2script_idx = lang2script_mat.argmax(dim=1)       

    p_m_all = torch.softmax(logits_m, dim=1)
    p_s_all = torch.softmax(logits_s, dim=1)
    p_l_all = torch.softmax(logits_l, dim=1)

    pred_l_idx = p_l_all.argmax(dim=1)
    pred_s_parent_idx = lang2script_idx[pred_l_idx]
    pred_m_grand_idx = script2macro_idx[pred_s_parent_idx]

    batch_p_l = p_l_all[torch.arange(p_l_all.size(0)), pred_l_idx]
    batch_p_s = p_s_all[torch.arange(p_s_all.size(0)), pred_s_parent_idx]
    batch_p_m = p_m_all[torch.arange(p_m_all.size(0)), pred_m_grand_idx]

    penalty_l_s = torch.clamp(batch_p_l - batch_p_s, min=0.0)
    penalty_s_m = torch.clamp(batch_p_s - batch_p_m, min=0.0)
    loss_violation = torch.mean(penalty_l_s + penalty_s_m)

    soft_script_target = torch.matmul(p_l_all, lang2script_mat)   
    soft_macro_target = torch.matmul(soft_script_target, script2macro_mat) 

    kl_loss_s = F.kl_div(torch.log_softmax(logits_s, dim=1), soft_script_target, reduction='batchmean')
    kl_loss_m = F.kl_div(torch.log_softmax(logits_m, dim=1), soft_macro_target, reduction='batchmean')
    loss_distill = kl_loss_s + kl_loss_m

    if log_vars_hcd is not None:
        clamped_log_vars = torch.clamp(log_vars_hcd, min=-2.0, max=5.0)
        precision_distill = torch.exp(-clamped_log_vars[0])
        precision_violation = torch.exp(-clamped_log_vars[1])
        weighted_distill = precision_distill * loss_distill + clamped_log_vars[0]
        weighted_violation = precision_violation * loss_violation + clamped_log_vars[1]
        loss_hcd = weighted_distill + weighted_violation
    else:
        loss_hcd = 0.2 * loss_distill + 0.1 * loss_violation

    with torch.no_grad():
        if log_vars_hcd is not None:
            clamped_vars = torch.clamp(log_vars_hcd, min=-2.0, max=5.0)
            actual_w_distill = torch.exp(-clamped_vars[0]).item()
            actual_w_violation = torch.exp(-clamped_vars[1]).item()
        else:
            actual_w_distill, actual_w_violation = 0.2, 0.1
    
    tree_loss_value = torch.tensor(0.0, device=device)
    if use_tree and tree_loss_fn is not None:
        sig_all = torch.cat([
            torch.sigmoid(logits_m),
            torch.sigmoid(logits_s),
            torch.sigmoid(logits_l)
        ], dim=1)
        tree_loss_value = tree_loss_fn(sig_all, target_sig, device)
        loss = loss_ce + tree_weight * tree_loss_value + loss_hcd
    else:
        loss = loss_ce + loss_hcd

    with torch.no_grad():
        actual_w_distill = torch.exp(-log_vars_hcd[0]).item() if log_vars_hcd is not None else 0.2
        actual_w_violation = torch.exp(-log_vars_hcd[1]).item() if log_vars_hcd is not None else 0.1

    loss_info = {
        "loss_m": float(loss_m.detach().cpu()),
        "loss_s": float(loss_s.detach().cpu()),
        "loss_l": float(loss_l.detach().cpu()),
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_tree": float(tree_loss_value.detach().cpu()),
        "loss_distill": float(loss_distill.detach().cpu()),
        "loss_violation": float(loss_violation.detach().cpu()),
        "w_m": float(ce_weights[0].detach().cpu()),
        "w_s": float(ce_weights[1].detach().cpu()),
        "w_l": float(ce_weights[2].detach().cpu()),
        "dynamic_w_distill": actual_w_distill,
        "dynamic_w_violation": actual_w_violation,
    }

    return loss, loss_info


def train_epoch(net, trainloader, optimizer, CELoss, tree_loss_fn,
                device, dataset_info, args, epoch):
    log_vars_hcd = optimizer.param_groups[-1]["params"][0]
    net.train()
    train_loss = 0.0
    all_preds, all_targets = [], []

    if args.lr_adjt == "Cos":
        lr_list = [0.002] * 9 + [0.0002]
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]["lr"] = cosine_anneal_schedule(
                epoch, args.epoch, lr_list[min(nlr, len(lr_list) - 1)]
            )

    num_m = dataset_info["num_macros"]
    num_s = dataset_info["num_scripts"]
    lang_offset = num_m + num_s

    loss_info_last = None

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits_m, logits_s, logits_l, feats = net(inputs)

        loss, loss_info = compute_training_loss(
            logits_m, logits_s, logits_l, targets,
            CELoss, device, dataset_info,
            use_adaptive_ce=args.use_adaptive_ce,
            adaptive_gamma=args.adaptive_gamma,
            tree_loss_fn=tree_loss_fn,
            use_tree=args.use_tree,
            tree_weight=args.tree_weight,
            log_vars_hcd=log_vars_hcd
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_info_last = loss_info

        with torch.no_grad():
            pred = logits_l.argmax(dim=1)
            all_preds.extend((pred + lang_offset).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    metrics = calculate_accuracy_metrics(all_preds, all_targets, dataset_info)
    h_f1, hp, hr = calculate_hierarchical_f1(all_preds, all_targets, dataset_info)
    sev = calculate_severity_of_mistake(all_preds, all_targets, dataset_info)

    metrics.update({
        "hierarchical_f1": round(h_f1, 4),
        "hierarchical_precision": round(hp, 4),
        "hierarchical_recall": round(hr, 4),
        "severity_of_mistake": round(sev, 4),
        "loss": round(train_loss / (batch_idx + 1), 4),
    })
    if loss_info_last is not None:
        metrics.update({k: round(v, 4) for k, v in loss_info_last.items()})

    return metrics


@torch.no_grad()
def test(net, testloader, CELoss, tree_loss_fn, device, dataset_info, args, calibrator=None, log_vars_hcd=None):
    net.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []
    all_confs = []

    num_m = dataset_info["num_macros"]
    num_s = dataset_info["num_scripts"]
    lang_offset = num_m + num_s

    loss_info_last = None

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits_m, logits_s, logits_l, feats = net(inputs)

        loss, loss_info = compute_training_loss(
            logits_m, logits_s, logits_l, targets,
            CELoss, device, dataset_info,
            use_adaptive_ce=args.use_adaptive_ce,
            adaptive_gamma=args.adaptive_gamma,
            tree_loss_fn=tree_loss_fn,
            use_tree=args.use_tree,
            tree_weight=args.tree_weight,
            log_vars_hcd=log_vars_hcd
        )

        test_loss += loss.item()
        loss_info_last = loss_info

        if calibrator is not None:
            final_lang_prob, pred, conf, aux = calibrator.predict(logits_m, logits_s, logits_l, feats)
        else:
            prob = torch.softmax(logits_l, dim=1)
            conf, pred = prob.max(dim=1)

        all_preds.extend((pred + lang_offset).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_confs.extend(conf.cpu().numpy())

    metrics = calculate_accuracy_metrics(all_preds, all_targets, dataset_info)
    h_f1, hp, hr = calculate_hierarchical_f1(all_preds, all_targets, dataset_info)
    sev = calculate_severity_of_mistake(all_preds, all_targets, dataset_info)

    metrics.update({
        "hierarchical_f1": round(h_f1, 4),
        "hierarchical_precision": round(hp, 4),
        "hierarchical_recall": round(hr, 4),
        "severity_of_mistake": round(sev, 4),
        "loss": round(test_loss / (batch_idx + 1), 4),
    })

    if loss_info_last is not None:
        metrics.update({k: round(v, 4) for k, v in loss_info_last.items()})

    # Reliability-aware metrics, analogous to corrected accuracy on high-confidence predictions.
    if calibrator is not None and args.conf_threshold > 0:
        conf_tensor = torch.tensor(all_confs)
        keep_mask = conf_tensor >= args.conf_threshold
        keep_indices = keep_mask.nonzero(as_tuple=False).view(-1).tolist()

        unreliable_rate = 100.0 * (1.0 - float(keep_mask.float().mean()))

        if len(keep_indices) > 0:
            conf_preds = [all_preds[i] for i in keep_indices]
            conf_targets = [all_targets[i] for i in keep_indices]
            conf_metrics = calculate_accuracy_metrics(conf_preds, conf_targets, dataset_info)
            conf_h_f1, _, _ = calculate_hierarchical_f1(conf_preds, conf_targets, dataset_info)
            metrics.update({
                "unreliable_rate": round(unreliable_rate, 2),
                "confident_accuracy": round(conf_metrics["accuracy"], 2),
                "confident_h_f1": round(conf_h_f1, 4),
                "num_confident": len(keep_indices),
            })
        else:
            metrics.update({
                "unreliable_rate": round(unreliable_rate, 2),
                "confident_accuracy": 0.0,
                "confident_h_f1": 0.0,
                "num_confident": 0,
            })

    return metrics


def main():
    args = arg_parse()
    torch.manual_seed(args.seed)

    save_name = f"hrn4clip_{args.backbone}_e{args.epoch}_b{args.batch}"
    if args.freeze_clip:
        save_name += "_frozen"
    if args.use_adaptive_ce:
        save_name += "_adaCE"
    if args.use_tree:
        save_name += f"_tree{args.tree_weight}"
    if args.use_conf_calib:
        save_name += "_confcalib"

    logger = Logger(args.save_dir, save_name)

    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    if args.backbone == 'RN50x64':
        target_size = 448
    elif args.backbone == 'RN50x4':
        target_size = 288
    elif args.backbone == 'RN50x16':
        target_size = 384
    else:
        target_size = 224

    transform_train = transforms.Compose([
        ResizeWithPadding(target_size),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.RandomRotation(degrees=3),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    transform_test = transforms.Compose([
        ResizeWithPadding(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    trainset = MultilangDataset(os.path.join(args.data_path, "train"), transform_train)
    testset = MultilangDataset(os.path.join(args.data_path, "test"), transform_test)
    dataset_info = trainset.get_hierarchy_info()

    print(f"Dataset: L1={dataset_info['num_macros']} L2={dataset_info['num_scripts']} "
          f"L3={dataset_info['num_languages']} | Total={dataset_info['total_nodes']} nodes")

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    device = torch.device(f"cuda:{args.device[0]}" if torch.cuda.is_available() else "cpu")

    net = build_net(args, dataset_info, device)

    if len(args.device) > 1:
        net = torch.nn.DataParallel(net, device_ids=list(map(int, args.device)))

    CELoss = nn.CrossEntropyLoss()

    tree = None
    if args.use_tree:
        tree = TreeLoss(
            dataset_info["trees"],
            dataset_info["total_nodes"],
            dataset_info["levels"],
            device,
        )

    model_for_params = unwrap_model(net)

    param_groups = [
        {"params": model_for_params.classifier_macro.parameters(), "lr": 0.002},
        {"params": model_for_params.classifier_script.parameters(), "lr": 0.002},
        {"params": model_for_params.classifier_language.parameters(), "lr": 0.002},
        {"params": model_for_params.macro_proj.parameters(), "lr": 0.002},
        {"params": model_for_params.script_proj.parameters(), "lr": 0.002},
        {"params": model_for_params.language_proj.parameters(), "lr": 0.002},
        {"params": model_for_params.macro_conv.parameters(), "lr": 0.002},
        {"params": model_for_params.script_conv.parameters(), "lr": 0.002},
        {"params": model_for_params.language_conv.parameters(), "lr": 0.002},
    ]

    if not args.freeze_clip:
        exclude_keywords = [
            'classifier_macro', 'classifier_script', 'classifier_language',
            'classifier_language_softmax', 'macro_proj', 'script_proj',
            'language_proj', 'macro_conv', 'script_conv', 'language_conv'
        ]
        if hasattr(net, 'attnpool'):
            exclude_keywords.append('attnpool')

        backbone_params = [p for n, p in net.named_parameters()
                           if not any(k in n for k in exclude_keywords)
                           and p.requires_grad]
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': 0.0002})

    log_vars_hcd = nn.Parameter(torch.tensor([0.0, 0.0], device=device, requires_grad=True))
    
    param_groups.append({"params": [log_vars_hcd], "lr": 0.002})

    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    total_params = sum(p.numel() for p in model_for_params.parameters())
    trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
    print(f"\nParameters: Total={total_params:,} | Trainable={trainable_params:,} "
          f"({'frozen' if args.freeze_clip else 'full'})")
    print(f"Optimizer groups: {len(param_groups)} | LR: {[g['lr'] for g in param_groups]}")
    print("-" * 80)

    calibrator = None
    if args.use_conf_calib:
        lang2script, script2macro = build_hierarchy_matrices(dataset_info, device)
        calibrator = HierarchicalConfidenceCalibrator(
            num_macros=dataset_info["num_macros"],
            num_scripts=dataset_info["num_scripts"],
            num_languages=dataset_info["num_languages"],
            lang2script=lang2script,
            script2macro=script2macro,
            feature_dim=512,
            beta=args.conf_beta,
            correct_only=True,
        )


    # Evaluation Only
    if args.eval_only:

        print("\n" + "=" * 80)
        print("Evaluation Only Mode")
        print("=" * 80)

        log_vars_hcd = nn.Parameter(
            torch.tensor([0.0, 0.0], device=device)
        )

        eval_calibrator = None

        if calibrator is not None:
            print("[Calib] Fitting hierarchical confidence calibrator...")

            calibrator.fit(
                net,
                trainloader,
                dataset_info,
                device
            )

            eval_calibrator = calibrator

        test_m = test(
            net,
            testloader,
            CELoss,
            tree,
            device,
            dataset_info,
            args,
            calibrator=eval_calibrator,
            log_vars_hcd=log_vars_hcd
        )

        print("\nTest Results")
        print("-" * 80)

        for k, v in test_m.items():
            print(f"{k}: {v}")

        return
    
    print(f"\nTraining {args.epoch} epochs...")
    print("-" * 80)

    for epoch in range(args.epoch):
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"] if args.lr_adjt == "Step" else optimizer.param_groups[0]["lr"]

        train_m = train_epoch(
            net, trainloader, optimizer, CELoss, tree,
            device, dataset_info, args, epoch
        )

        eval_calibrator = None
        if calibrator is not None and epoch >= args.calib_start_epoch:
            print("[Calib] Fitting hierarchical confidence calibrator on training embeddings...")
            calibrator.fit(net, trainloader, dataset_info, device)
            eval_calibrator = calibrator

        test_m = test(
            net, testloader, CELoss, tree,
            device, dataset_info, args,
            calibrator=eval_calibrator,
            log_vars_hcd=log_vars_hcd
        )

        epoch_time = time.time() - start

        is_best = logger.log_and_print(epoch, train_m, test_m, epoch_time, current_lr)
        if is_best:
            logger.update_best(epoch, test_m["hierarchical_f1"])
            save_checkpoint(net, optimizer, epoch, args.save_dir, save_name)

        if args.lr_adjt == "Step":
            scheduler.step()

    logger.print_final_summary()


if __name__ == "__main__":
    main()
