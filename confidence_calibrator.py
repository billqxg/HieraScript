import torch
import torch.nn.functional as F

from utils_multilang import get_hierarchy_targets


class HierarchicalConfidenceCalibrator:
    def __init__(
        self,
        num_macros,
        num_scripts,
        num_languages,
        lang2script,
        script2macro,
        feature_dim=512,
        beta=10.0,
        eps=1e-8,
        correct_only=True,
    ):
        self.num_macros = num_macros
        self.num_scripts = num_scripts
        self.num_languages = num_languages
        self.lang2script = lang2script
        self.script2macro = script2macro
        self.feature_dim = feature_dim
        self.beta = beta
        self.eps = eps
        self.correct_only = correct_only

        self.proto_m = None
        self.proto_s = None
        self.proto_l = None

    @torch.no_grad()
    def fit(self, model, dataloader, dataset_info, device):
        model.eval()

        all_sum_m = torch.zeros(self.num_macros, self.feature_dim, device=device)
        all_cnt_m = torch.zeros(self.num_macros, device=device)

        all_sum_s = torch.zeros(self.num_scripts, self.feature_dim, device=device)
        all_cnt_s = torch.zeros(self.num_scripts, device=device)

        all_sum_l = torch.zeros(self.num_languages, self.feature_dim, device=device)
        all_cnt_l = torch.zeros(self.num_languages, device=device)

        cor_sum_m = torch.zeros_like(all_sum_m)
        cor_cnt_m = torch.zeros_like(all_cnt_m)
        cor_sum_s = torch.zeros_like(all_sum_s)
        cor_cnt_s = torch.zeros_like(all_cnt_s)
        cor_sum_l = torch.zeros_like(all_sum_l)
        cor_cnt_l = torch.zeros_like(all_cnt_l)

        num_m = dataset_info["num_macros"]
        num_s = dataset_info["num_scripts"]
        lang_offset = num_m + num_s

        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits_m, logits_s, logits_l, feats = model(images)

            macro_t, script_t, _ = get_hierarchy_targets(targets, device, dataset_info)
            lang_t = targets - lang_offset

            fm = F.normalize(feats["macro"], dim=-1)
            fs = F.normalize(feats["script"], dim=-1)
            fl = F.normalize(feats["language"], dim=-1)

            pred_m = logits_m.argmax(dim=1)
            pred_s = logits_s.argmax(dim=1)
            pred_l = logits_l.argmax(dim=1)

            for i in range(images.size(0)):
                m = int(macro_t[i].item())
                s = int(script_t[i].item())
                l = int(lang_t[i].item())

                all_sum_m[m] += fm[i]
                all_cnt_m[m] += 1
                all_sum_s[s] += fs[i]
                all_cnt_s[s] += 1
                all_sum_l[l] += fl[i]
                all_cnt_l[l] += 1

                if pred_m[i].item() == m:
                    cor_sum_m[m] += fm[i]
                    cor_cnt_m[m] += 1
                if pred_s[i].item() == s:
                    cor_sum_s[s] += fs[i]
                    cor_cnt_s[s] += 1
                if pred_l[i].item() == l:
                    cor_sum_l[l] += fl[i]
                    cor_cnt_l[l] += 1

        self.proto_m = self._finalize_prototypes(cor_sum_m, cor_cnt_m, all_sum_m, all_cnt_m)
        self.proto_s = self._finalize_prototypes(cor_sum_s, cor_cnt_s, all_sum_s, all_cnt_s)
        self.proto_l = self._finalize_prototypes(cor_sum_l, cor_cnt_l, all_sum_l, all_cnt_l)

        return self

    def _finalize_prototypes(self, cor_sum, cor_cnt, all_sum, all_cnt):
        if self.correct_only:
            use_correct = cor_cnt > 0
            sums = torch.where(use_correct[:, None], cor_sum, all_sum)
            cnts = torch.where(use_correct, cor_cnt, all_cnt)
        else:
            sums = all_sum
            cnts = all_cnt

        proto = sums / cnts.clamp_min(1.0).unsqueeze(1)
        proto = F.normalize(proto, dim=-1)
        return proto

    def _calibrate_one_level(self, logits, feat, proto):
        p = F.softmax(logits, dim=1)
        feat = F.normalize(feat, dim=-1)

        sim = feat @ proto.t()
        dist = 1.0 - sim

        confidence = torch.exp(-self.beta * dist)
        p_cal = p * confidence
        p_cal = p_cal / p_cal.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return p_cal, confidence

    @torch.no_grad()
    def predict(self, logits_m, logits_s, logits_l, feats):
        if self.proto_m is None or self.proto_s is None or self.proto_l is None:
            raise RuntimeError("Calibrator has not been fitted. Call calibrator.fit(...) first.")

        p_m, conf_m = self._calibrate_one_level(logits_m, feats["macro"], self.proto_m)
        p_s, conf_s = self._calibrate_one_level(logits_s, feats["script"], self.proto_s)
        p_l, conf_l = self._calibrate_one_level(logits_l, feats["language"], self.proto_l)

        lang2script_idx = self.lang2script.argmax(dim=1)       # [num_languages]
        script2macro_idx = self.script2macro.argmax(dim=1)     # [num_scripts]
        lang2macro_idx = script2macro_idx[lang2script_idx]     # [num_languages]

        parent_script_prob = p_s[:, lang2script_idx]           # [B, num_languages]
        parent_macro_prob = p_m[:, lang2macro_idx]             # [B, num_languages]

        final_lang_score = p_l * parent_script_prob * parent_macro_prob
        final_lang_prob = final_lang_score / final_lang_score.sum(dim=1, keepdim=True).clamp_min(self.eps)

        final_conf, final_pred = final_lang_prob.max(dim=1)

        aux = {
            "p_macro_cal": p_m,
            "p_script_cal": p_s,
            "p_language_cal": p_l,
            "conf_macro_proto": conf_m,
            "conf_script_proto": conf_s,
            "conf_language_proto": conf_l,
        }

        return final_lang_prob, final_pred, final_conf, aux
