import numpy as np
import torch
import random
from collections import defaultdict

DEBUG_MODE = False
DEBUG_SAMPLE_LIMIT = 10


def get_hierarchy_targets(targets, device, dataset_info):
    num_macros = dataset_info['num_macros']
    num_scripts = dataset_info['num_scripts']
    
    macro_targets = []
    script_targets = []
    target_list_sig = []
    
    macro_max = num_macros
    script_min = num_macros
    script_max = num_macros + num_scripts
    language_min = num_macros + num_scripts
    
    for i in range(targets.size(0)):
        target = targets[i].item()
        
        if target < macro_max:
            # Level 1: Macro
            macro_targets.append(int(target))
            script_targets.append(-1)
        elif target < script_max:
            # Level 2: Script
            script_idx = int(target - script_min)
            macro_idx = _find_parent_macro(target, dataset_info)
            macro_targets.append(macro_idx)
            script_targets.append(script_idx)
        else:
            # Level 3: Language
            lang_idx = int(target - language_min)
            script_idx, macro_idx = _find_parents(target, dataset_info)
            macro_targets.append(macro_idx)
            script_targets.append(script_idx)
        
        target_list_sig.append(int(target))
    
    macro_targets = torch.from_numpy(np.array(macro_targets)).to(device)
    script_targets = torch.from_numpy(np.array(script_targets)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    
    return macro_targets, script_targets, target_list_sig


def _find_parent_macro(script_id, dataset_info):
    trees = dataset_info['trees']  # [leaf_id, script_id, macro_id]
    
    for tree in trees:
        if tree[1] == script_id:
            return tree[2]
    return 0


def _find_parents(lang_id, dataset_info):
    trees = dataset_info['trees']
    
    for tree in trees:
        if tree[0] == lang_id:
            script_idx = tree[1] - dataset_info['num_macros']
            macro_idx = tree[2]
            return script_idx, macro_idx
    
    return 0, 0


def calculate_hierarchical_f1(predictions, targets, dataset_info, verbose=False):
    dataset = dataset_info['dataset']
    
    total_hp = 0.0
    total_hr = 0.0
    n = len(predictions)
    
    debug_stats = {
        'correct': 0,
        'same_macro': 0,
        'diff_macro': 0,
        'total': n
    }
    
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        anc_pred = dataset.get_ancestors(int(pred))
        anc_target = dataset.get_ancestors(int(target))
        intersection = len(anc_pred & anc_target)
        
        hp = intersection / len(anc_pred) if len(anc_pred) > 0 else 0
        hr = intersection / len(anc_target) if len(anc_target) > 0 else 0
        total_hp += hp
        total_hr += hr
        
        if pred == target:
            debug_stats['correct'] += 1
        else:
            pred_macro = _get_macro_id(int(pred), dataset_info)
            target_macro = _get_macro_id(int(target), dataset_info)
            if pred_macro == target_macro:
                debug_stats['same_macro'] += 1
            else:
                debug_stats['diff_macro'] += 1
    
    avg_hp = total_hp / n
    avg_hr = total_hr / n
    
    # F1
    if avg_hp + avg_hr == 0:
        h_f1 = 0.0
    else:
        h_f1 = 2 * avg_hp * avg_hr / (avg_hp + avg_hr)
    
    if DEBUG_MODE or verbose:
        print("\n" + "="*60)
        print("层级F1调试信息:")
        print(f"  总样本: {n}")
        print(f"  正确: {debug_stats['correct']} ({100*debug_stats['correct']/n:.1f}%)")
        print(f"  错误(同Macro): {debug_stats['same_macro']} ({100*debug_stats['same_macro']/n:.1f}%)")
        print(f"  错误(跨Macro): {debug_stats['diff_macro']} ({100*debug_stats['diff_macro']/n:.1f}%)")
        print(f"  HP: {avg_hp:.4f}, HR: {avg_hr:.4f}, H-F1: {h_f1:.4f}")
        print("="*60)
    
    return h_f1, avg_hp, avg_hr


def _get_macro_id(node_id, dataset_info):
    if node_id == -1:
        return -1
    elif node_id < dataset_info['num_macros']:
        return node_id
    elif node_id < dataset_info['num_macros'] + dataset_info['num_scripts']:
        return _find_parent_macro(node_id, dataset_info)
    else:
        _, macro_idx = _find_parents(node_id, dataset_info)
        return macro_idx


def calculate_severity_of_mistake(predictions, targets, dataset_info, epoch=0, split="test"):
    dataset = dataset_info['dataset']
    
    def get_height(node):
        if node == -1:
            return 3  # root
        elif node < dataset_info['num_macros']:
            return 2  # Macro
        elif node < dataset_info['num_macros'] + dataset_info['num_scripts']:
            return 1  # Script
        else:
            return 0  # Language
    
    def find_lca(node1, node2):
        ancestors1 = dataset.get_ancestors(node1)
        ancestors2 = dataset.get_ancestors(node2)
        
        common_ancestors = ancestors1 & ancestors2
        
        if not common_ancestors:
            return -1
        
        lca = min(common_ancestors, key=lambda x: get_height(x))
        return lca
    
    error_stats = {
        'total_errors': 0,
        'lca_is_root': 0,        # LCA is root
        'lca_is_macro': 0,       # LCA is Macro
        'lca_is_script': 0,      # LCA is Script
        'correct': 0,
        'severity_sum': 0.0
    }
    
    total_severity = 0.0
    n = len(predictions)
    correct = 0
    
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        pred_int = int(pred)
        target_int = int(target)
        
        if pred_int == target_int:
            correct += 1
            error_stats['correct'] += 1
            continue
        
        error_stats['total_errors'] += 1
        
        lca = find_lca(pred_int, target_int)
        height_lca = get_height(lca)
        height_pred = get_height(pred_int)
        height_target = get_height(target_int)
        height_root = 3
        
        if lca == -1:
            error_stats['lca_is_root'] += 1
        elif height_lca == 2:
            error_stats['lca_is_macro'] += 1
        elif height_lca == 1:
            error_stats['lca_is_script'] += 1
        
        numerator = 2 * height_lca - (height_target + height_pred)
        denominator = 2 * height_root - (height_target + height_pred)
        
        severity = numerator / denominator if denominator != 0 else 0.0
        total_severity += severity
        error_stats['severity_sum'] += severity
    
    error_count = n - correct
    avg_severity = total_severity / error_count if error_count > 0 else 0.0
    
    if DEBUG_MODE and error_count > 0:
        print(f"\n错误严重性分析 [{split}] [轮次 {epoch}]")
        print(f"总样本: {n}, 正确: {correct}, 错误: {error_count}")
        print(f"错误分布: Root={error_stats['lca_is_root']}, "
              f"Macro={error_stats['lca_is_macro']}, "
              f"Script={error_stats['lca_is_script']}")
        print(f"平均严重性: {avg_severity:.4f} (0=无错, 1=最严重)")
    
    return avg_severity


def calculate_accuracy_metrics(predictions, targets, dataset_info):
    num_macros = dataset_info['num_macros']
    num_scripts = dataset_info['num_scripts']
    lang_threshold = num_macros + num_scripts
    
    macro_correct = 0
    script_correct = 0
    lang_correct = 0
    
    macro_total = 0
    script_total = 0
    lang_total = 0
    
    for pred, target in zip(predictions, targets):
        pred = int(pred)
        target = int(target)
        
        if target < num_macros:
            # Level 1: Macro
            macro_total += 1
            if pred == target:
                macro_correct += 1
        elif target < lang_threshold:
            # Level 2: Script
            script_total += 1
            macro_total += 1
            
            if pred == target:
                script_correct += 1
            
            _, target_macro = _find_parents_for_eval(target, dataset_info)
            _, pred_macro = _find_parents_for_eval(pred, dataset_info)
            if pred_macro == target_macro:
                macro_correct += 1
        else:
            # Level 3: Language
            lang_total += 1
            script_total += 1
            macro_total += 1
            
            if pred == target:
                lang_correct += 1
            
            target_script, target_macro = _find_parents_for_eval(target, dataset_info)
            pred_script, pred_macro = _find_parents_for_eval(pred, dataset_info)
            
            if pred_script == target_script:
                script_correct += 1
            if pred_macro == target_macro:
                macro_correct += 1
    
    metrics = {
        'accuracy': 100.0 * sum([p==t for p,t in zip(predictions, targets)]) / len(predictions),
        'macro_acc': 100.0 * macro_correct / max(macro_total, 1),
        'script_acc': 100.0 * script_correct / max(script_total, 1),
        'language_acc': 100.0 * lang_correct / max(lang_total, 1)
    }
    
    if DEBUG_MODE:
        print(f"\n[准确率调试]")
        print(f"Macro级: {macro_correct}/{macro_total} = {metrics['macro_acc']:.1f}%")
        print(f"Script级: {script_correct}/{script_total} = {metrics['script_acc']:.1f}%")
        print(f"Language级: {lang_correct}/{lang_total} = {metrics['language_acc']:.1f}%")
    
    return metrics


def _find_parents_for_eval(node_id, dataset_info):
    num_macros = dataset_info['num_macros']
    num_scripts = dataset_info['num_scripts']
    lang_threshold = num_macros + num_scripts
    
    if node_id < num_macros:
        return -1, -1
    elif node_id < lang_threshold:
        script_idx = node_id - num_macros
        macro_idx = _find_parent_macro(node_id, dataset_info)
        return script_idx, macro_idx
    else:
        return _find_parents(node_id, dataset_info)


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % nb_epoch)
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


def build_hierarchy_matrices(dataset_info, device):
    num_m = dataset_info['num_macros']
    num_s = dataset_info['num_scripts']
    num_l = dataset_info['num_languages']

    lang2script = torch.zeros(num_l, num_s, device=device)
    script2macro = torch.zeros(num_s, num_m, device=device)

    for lang_id, script_id, macro_id in dataset_info['trees']:
        lang_idx = int(lang_id - num_m - num_s)
        script_idx = int(script_id - num_m)
        macro_idx = int(macro_id)

        if 0 <= lang_idx < num_l and 0 <= script_idx < num_s:
            lang2script[lang_idx, script_idx] = 1.0
        if 0 <= script_idx < num_s and 0 <= macro_idx < num_m:
            script2macro[script_idx, macro_idx] = 1.0

    return lang2script, script2macro
