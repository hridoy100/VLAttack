import copy
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd
import utils
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords
from filter_words import filter_words

nltk.download("stopwords")
config_atk = BertConfig.from_pretrained("bert-base-uncased")
filter_words = filter_words + stopwords.words("english") + ["?", "."]


class Feature(object):
    def __init__(self, seq_a):
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []


class Adv_attack:
    """
    Joint multimodal attack that shares a single budget between image perturbations
    and text edits. Image updates use PGD (BSA loss); text updates use masked LM
    substitutions filtered by semantic similarity. A Lagrange-style update keeps
    the combined budget (normalized image delta + normalized edit count) near 1.
    """

    def __init__(
        self,
        vqa_model,
        pretrain_model,
        tokenizer,
        device,
        correct_idx_list,
        correct_pred_list,
        USE_model,
        joint_cfg=None,
    ):
        # default joint constraints
        self.joint_cfg = {
            "epsilon": 0.125,
            "alpha": 0.01,
            "steps": 40,
            "max_edits": 3,
            "lambda_init": 0.1,
            "rho": 0.5,
            "patience": 5,
            "modality_switch_prob": 0.25,
            "max_text_candidates": 12,
        }
        if joint_cfg:
            self.joint_cfg.update(joint_cfg)

        self.attack_dict = {}
        self.text_budget = 100000
        self.cos_sim = 0.95
        self.k = 10
        self.tokenizer = tokenizer
        self.tokenizer_mlm = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case="uncased" in "bert-base-uncased"
        )

        self.correct_list = correct_idx_list
        self.blip_ans_table = correct_pred_list
        self.white_model = pretrain_model
        self.black_model = vqa_model
        self.USE_model = USE_model
        self.device = device
        self.acc_list = []
        self.mlm_model = BertForMaskedLM.from_pretrained(
            "bert-base-uncased", config=config_atk
        ).to(self.device)

    def Gen_ori_feats(self, batch):
        image = batch["image"].to(self.device, non_blocking=True)
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(
            image, batch["question"][0]
        )
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return img_feats, txt_feats

    def pgd_attack(self, x):
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(
            x, self.batch["question"]
        )
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats, img_feats]

    def _tokenize(self, seq, tokenizer):
        seq = seq.replace("\n", "").lower()
        words = seq.split(" ")
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i
        c_loss = torch.nn.CrossEntropyLoss(reduction="none")
        all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(
        self, substitutes, tokenizer, mlm_model, substitutes_score=None, use_bpe=True, threshold=0.3
    ):
        words = []
        sub_len, _ = substitutes.size()  # sub-len, k
        if sub_len == 0:
            return words
        if sub_len == 1:
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
        return words

    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ["[MASK]"] + words[i + 1 :])
        return masked_words

    def black_box_predict(self, image, text):
        answer_ids, _, _ = self.black_model(
            image, text, self.answer_candidates, train=False, inference="rank", k_test=128
        )
        out_v = []
        for answer_id in answer_ids:
            out_v.append({"answer": self.answer_list[answer_id]})
        return out_v[0]["answer"]

    def get_important_scores(self, words, batch, tgt_pos, score, image):
        masked_words = self._get_masked(words)
        texts = [" ".join(words) for words in masked_words]
        important_scores = []
        for mlm in texts:
            _, topk_ids, topk_probs = self.black_model(
                image, mlm, self.answer_candidates, train=False, inference="rank", k_test=128
            )
            _, pred = topk_probs[0].max(dim=0)
            if tgt_pos not in list(topk_ids[0].cpu().numpy()):
                important_scores.append((torch.tensor(-10000).to(self.device)).data.cpu().numpy())
            else:
                im_value = topk_probs[0][torch.where(topk_ids[0] == tgt_pos)][0]
                important_scores.append((im_value - score).data.cpu().numpy())
        return np.array(important_scores)

    def build_text_bank(self, batch, tgt_pos, score, gth):
        ori_text = batch["question"][0]
        image = batch["image"].to(self.device, non_blocking=True)
        feature = Feature(ori_text.lower())
        tokenizer = self.tokenizer_mlm
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        max_length = 512
        inputs = tokenizer.encode_plus(
            feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True
        )
        sub_words = ["[CLS]"] + sub_words[:2] + sub_words[2 : max_length - 2] + ["[SEP]"]
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)
        word_predictions = word_predictions[1 : len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1 : len(sub_words) + 1, :]
        important_scores = self.get_important_scores(words, batch, tgt_pos, score, image)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=False)
        final_words = copy.deepcopy(words)
        text_bank = []
        sim_list = []
        success = False
        for top_index in list_of_index:
            if len(text_bank) >= self.joint_cfg["max_text_candidates"]:
                break
            tgt_word = words[top_index[0]]
            if tgt_word in filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue
            substitutes = word_predictions[keys[top_index[0]][0] : keys[top_index[0]][1]]
            word_pred_scores = word_pred_scores_all[
                keys[top_index[0]][0] : keys[top_index[0]][1]
            ]
            substitutes = self.get_substitues(
                substitutes, tokenizer, self.mlm_model, substitutes_score=word_pred_scores
            )
            for substitute in substitutes:
                if substitute == tgt_word:
                    continue
                if "##" in substitute or substitute in filter_words:
                    continue
                temp_replace = copy.deepcopy(final_words)
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                embs = self.USE_model([ori_text, temp_text]).numpy()
                norm = np.linalg.norm(embs, axis=1)
                embs = embs / norm[:, None]
                sim = (embs[:1] * embs[1:]).sum(axis=1)[0]
                if sim < self.cos_sim:
                    continue
                sim_list.append(sim)
                text_bank.append(temp_text)
                ans_after_attack = self.black_box_predict(image, temp_text)
                if ans_after_attack != gth:
                    success = True
                    return [temp_text], success
        ordered = []
        sim_copy = copy.deepcopy(sim_list)
        for _ in range(len(sim_copy)):
            si = sim_copy.index(max(sim_copy))
            ordered.append(text_bank[si])
            sim_copy[si] = -1e8
        return ordered, success

    def current_img_cost(self, adv_img, ori_img):
        delta = torch.max(torch.abs(adv_img.detach() - ori_img.detach())).item()
        return min(1.0, delta / max(self.joint_cfg["epsilon"], 1e-12))

    @torch.no_grad()
    def evaluate(self, data_loader, tokenizer):
        answer_list = data_loader.dataset.answer_list
        self.answer_list = answer_list
        answer_candidates = self.black_model.tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)
        self.answer_candidates = answer_candidates
        answer_candidates.input_ids[:, 0] = self.black_model.tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        print_freq = 50000
        for _, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if len(self.acc_list) >= 5000:
                break
            if int(batch["question_id"][0]) not in self.correct_list:
                continue
            self.batch = copy.deepcopy(batch)
            ori_img = batch["image"].to(self.device, non_blocking=True)
            gth_ans = self.blip_ans_table[str(int(batch["question_id"][0]))]
            pred_ans = self.black_box_predict(ori_img, batch["question"][0])
            if pred_ans != gth_ans:
                continue

            ori_img_feats, ori_txt_feats = self.Gen_ori_feats(batch)
            answer_ids, topk_ids, topk_probs = self.black_model(
                ori_img, batch["question"][0], answer_candidates, train=False, inference="rank", k_test=128
            )
            score, pred = topk_probs[0].max(dim=0)
            tgt_pos = topk_ids[0][pred]

            text_bank, text_success = self.build_text_bank(batch, tgt_pos, score, gth_ans)
            if text_success:
                self.acc_list.append(1)
                if len(self.acc_list) % 100 == 0:
                    print(f"ASR of {str(len(self.acc_list))} samples:", sum(self.acc_list) / len(self.acc_list))
                continue

            adv_img = copy.deepcopy(ori_img)
            adv_text = batch["question"][0]
            text_edits = 0
            lambda_coeff = self.joint_cfg["lambda_init"]
            rho = self.joint_cfg["rho"]
            patience = self.joint_cfg["patience"]
            plateau = 0
            best_loss = None
            pgd_time = 0

            for step in range(self.joint_cfg["steps"]):
                img_cost = self.current_img_cost(adv_img, ori_img)
                text_cost = text_edits / max(1, self.joint_cfg["max_edits"])
                budget_violation = max(0.0, img_cost + text_cost - 1.0)
                lambda_coeff += rho * budget_violation

                choose_text = (
                    text_edits < self.joint_cfg["max_edits"]
                    and len(text_bank) > 0
                    and (random.random() < self.joint_cfg["modality_switch_prob"] or img_cost >= 1.0 - text_cost)
                )

                if choose_text:
                    if text_cost >= 1.0:
                        continue
                    adv_text = text_bank.pop(0)
                    text_edits += 1
                else:
                    remaining_budget = max(0.0, 1.0 - text_cost)
                    if remaining_budget <= 0:
                        continue
                    eps_rem = self.joint_cfg["epsilon"] * remaining_budget
                    step_size = min(self.joint_cfg["alpha"], eps_rem)
                    torch.set_grad_enabled(True)
                    adv_img, loss_list = pgd.projected_gradient_descent(
                        self.pgd_attack,
                        adv_img,
                        eps_rem,
                        step_size,
                        1,
                        np.inf,
                        clip_min=-1.0,
                        clip_max=1.0,
                        y=[ori_txt_feats, ori_img_feats, None, None, None],
                        time=pgd_time,
                        ori_x=ori_img,
                        method="BSA",
                    )
                    torch.set_grad_enabled(False)
                    pgd_time = 1
                    current_loss = loss_list[-1] if isinstance(loss_list, list) else None
                    if current_loss is not None:
                        if best_loss is None or current_loss > best_loss + 1e-4:
                            best_loss = current_loss
                            plateau = 0
                        else:
                            plateau += 1

                pred_ans = self.black_box_predict(adv_img, adv_text)
                if pred_ans != gth_ans:
                    self.acc_list.append(1)
                    break
                if plateau >= patience:
                    if len(text_bank) == 0:
                        break
                    plateau = 0
            else:
                self.acc_list.append(0)

            if len(self.acc_list) % 100 == 0 and len(self.acc_list) != 0:
                print(f"ASR of {str(len(self.acc_list))} samples:", sum(self.acc_list) / len(self.acc_list))
        print("ASR: ", sum(self.acc_list) / len(self.acc_list))
