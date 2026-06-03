import random
from collections import Counter
import numpy as np


class MajorityVoter:
    def __init__(self, source_num):
        self.source_num = source_num

    def prepare_for_fusion(self, cand_answer):
        self.ans_set = []
        for src, pairs in cand_answer.items():
            for ans, _ in pairs:
                self.ans_set.append(ans)
        self.counter = Counter(self.ans_set)

    def fusion_for_case(self, ans_set, inv_aid, topk):
        counter = Counter(ans_set)
        topk_ans = counter.most_common(topk)
        case_id = []
        for ans, _ in topk_ans:
            case_id.extend(inv_aid[ans])
        return case_id

    def iterate_fusion(self, **kwargs):
        ans_len = len(self.ans_set)
        veracity = {a: c / ans_len for a, c in self.counter.items()}
        self.veracity = np.array([veracity[a] for a in self.ans_set])
        return kwargs["threshold"]


class DARTFusion:
    def __init__(self, source_num, init_veracity, rec_prior, sp_prior, max_iters):
        self.source_num = source_num
        self.init_veracity = init_veracity
        self.rec_prior = rec_prior
        self.sp_prior = sp_prior
        self.max_iters = max_iters

    def prepare_for_fusion(self, cand_answer):
        a_id, s_id = 0, 0
        self.ans_set = []
        src_ans_dict = {}
        src_w_ans = {}
        for src, pairs in cand_answer.items():
            for ans, match_score in pairs:
                self.ans_set.append(ans)
                if s_id in src_ans_dict:
                    src_ans_dict[s_id].append(a_id)
                else:
                    src_ans_dict[s_id] = [a_id]
                if ans in src_w_ans:
                    src_w_ans[ans].append(s_id)
                else:
                    src_w_ans[ans] = [s_id]
                a_id += 1
            if len(pairs) > 0:
                s_id += 1
        self.source_num = s_id
        self.veracity = np.array([self.init_veracity] * a_id)
        self.trust_rec = np.full((self.source_num, 1), self.rec_prior)
        self.trust_sp = np.full((self.source_num, 1), self.sp_prior)
        self.sa_mask = np.full((self.source_num, a_id), False)
        self.as_mask = np.full((self.source_num, a_id), False)
        for sid, ans_lst in src_ans_dict.items():
            self.sa_mask[sid, ans_lst] = True
        for aid, ans in enumerate(self.ans_set):
            src_lst = src_w_ans[ans]
            self.as_mask[src_lst, aid] = True

        supp_case = (
            1.0 - (~self.sa_mask).sum(axis=1) / pow(a_id, 2)
        ) / self.sa_mask.sum(axis=1)
        oppo_case = 1 / max(pow(a_id, 2), 1)
        self.conf = np.where(self.sa_mask, supp_case[:, np.newaxis], oppo_case)

    def update_trustworthiness(self):
        rec = np.where(self.sa_mask, self.veracity[np.newaxis, :], 0.0)
        sp = np.where(~self.sa_mask, 1.0 - self.veracity[np.newaxis, :], 0.0)
        self.trust_rec = rec.sum(axis=-1, keepdims=True) / self.sa_mask.sum(
            axis=-1, keepdims=True
        )
        self.trust_sp = sp.sum(axis=-1, keepdims=True) / (~self.sa_mask).sum(
            axis=-1, keepdims=True
        )

    def update_veracity(self):
        pos = np.where(self.as_mask, self.trust_rec, 1.0 - self.trust_sp)
        neg = np.where(~self.as_mask, self.trust_sp, 1.0 - self.trust_rec)
        pos = np.prod(np.power(pos, self.conf), axis=0)
        neg = np.prod(np.power(neg, self.conf), axis=0)
        self.veracity = ((1.0 - self.veracity) / self.veracity) * (neg / pos)
        self.veracity = 1.0 / (1 + self.veracity)

    def iterate_fusion(self, **kwargs):
        for its in range(self.max_iters):
            self.update_veracity()
            self.update_trustworthiness()
        return kwargs["threshold"]


class TruthFinder:
    def __init__(
        self,
        source_num,
        init_trust=0.9,
        gamma=0.3,
        rho=0.5,
        max_iters=10,
        early_stop=1e-3,
    ):
        self.source_num = source_num
        self.init_trust = init_trust
        self.max_iters = max_iters
        self.gamma = gamma
        self.rho = rho
        self.early_stop = early_stop

    def prepare_for_fusion(self, cand_answer):
        a_id, s_id = 0, 0
        self.ans_set = []
        src_ans_dict = {}
        src_w_ans = {}
        for src, pairs in cand_answer.items():
            for ans, match_score in pairs:
                self.ans_set.append(ans)
                if s_id in src_ans_dict:
                    src_ans_dict[s_id].append(a_id)
                else:
                    src_ans_dict[s_id] = [a_id]
                if ans in src_w_ans:
                    src_w_ans[ans].append(s_id)
                else:
                    src_w_ans[ans] = [s_id]
                a_id += 1
            if len(pairs) > 0:
                s_id += 1
        self.source_num = s_id
        self.sa_mask = np.full((self.source_num, a_id), False)
        self.as_mask = np.full((self.source_num, a_id), False)
        for sid, ans_lst in src_ans_dict.items():
            self.sa_mask[sid, ans_lst] = True
        for aid, ans in enumerate(self.ans_set):
            src_lst = src_w_ans[ans]
            self.as_mask[src_lst, aid] = True
        self.A = np.where(
            self.sa_mask, 1.0 / self.sa_mask.sum(axis=-1, keepdims=True), 0.0
        )
        # self.B = np.where(self.as_mask, self.rho * .5, 0) + \
        #          np.where(self.sa_mask, 1. - self.rho * .5, 0.)
        self.B = self.precalculate_B(a_id, self.rho)
        self.src_trust = np.full((self.source_num), self.init_trust)

    def precalculate_B(self, val_num, rho):
        B = np.eye(val_num, dtype=float)
        for aid1 in range(val_num):
            for aid2 in range(aid1, val_num):
                imp = self.dice_dist(self.ans_set[aid1], self.ans_set[aid2]) - 0.5
                B[aid1, aid2] = rho * imp
        B = B @ self.sa_mask.astype(float).T
        return B

    def dice_dist(self, str1, str2):
        # Sørensen–Dice on token multisets. Vendored upstream formula
        # divides by ``len(str1) + len(str2)`` which raises
        # ZeroDivisionError when both inputs are empty (observed in
        # the BoB products run: an empty candidate answer paired with
        # itself). By the standard set-similarity convention, two
        # empty sets are identical → similarity = 1.0; an empty paired
        # with a non-empty has no shared tokens → 0.0.
        if len(str1) == 0 and len(str2) == 0:
            return 1.0
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
        cnt1 = Counter(str1)
        cnt2 = Counter(str2)
        unions = cnt1 & cnt2
        return 2 * sum(unions.values()) / (len(str1) + len(str2))

    def cos_sim(self, vec_A, vec_B):
        norm_A = np.linalg.norm(vec_A)
        norm_B = np.linalg.norm(vec_B)
        sim_val = vec_A.dot(vec_B) / (norm_A * norm_B)
        return sim_val

    def iterate_fusion(self, **kwargs):
        its = 0
        while True:
            its += 1
            tau = -np.log(1.0 - self.src_trust)
            sigma = self.B @ tau
            self.veracity = 1.0 / (1.0 + np.exp(-self.gamma * sigma))
            trust_old = self.src_trust
            self.src_trust = self.A @ self.veracity
            if (
                self.cos_sim(trust_old, self.src_trust) >= (1.0 - self.early_stop)
                or its == self.max_iters
            ):
                break
        return kwargs["threshold"]


class CASEFusion:
    def __init__(self, source_num, dimension=10, **kwargs):
        self.source_num = source_num
        self.dimension = dimension
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.lr = kwargs["lr"]
        self.converge_rate = kwargs["converge_rate"]
        self.max_iters = kwargs["max_iters"]
        self.voter = MajorityVoter(source_num)

    def prepare_for_fusion(self, cand_answer):
        a_id, s_id = 0, 0
        self.ans_set = []
        self.scgraph = []
        src_w_ans = {}
        inv_ans_id = {}
        for src, pairs in cand_answer.items():
            for ans, match_score in pairs:
                self.ans_set.append(ans)
                self.scgraph.append((s_id, a_id))
                if s_id in src_w_ans:
                    src_w_ans[s_id].append(ans)
                else:
                    src_w_ans[s_id] = [ans]
                if ans in inv_ans_id:
                    inv_ans_id[ans].append(a_id)
                else:
                    inv_ans_id[ans] = [a_id]
                a_id += 1
            if len(pairs) > 0:
                s_id += 1
        self.source_num = s_id
        self.sa_mask = np.full((self.source_num, a_id), False)
        self.as_mask = np.full((self.source_num, a_id), False)
        for sid, aid in self.scgraph:
            self.sa_mask[sid, aid] = True

        self.src_emb = np.random.rand(self.source_num, self.dimension)
        self.ans_emb = np.random.rand(a_id, self.dimension)
        self.ss_same = self.sa_mask.astype(int) @ self.sa_mask.astype(int).T
        for src_a, alst_a in src_w_ans.items():
            for src_b, alst_b in src_w_ans.items():
                self.ss_same[src_a, src_b] = len(set(alst_a) & set(alst_b))
        self.ss_diff = np.stack([np.diag(self.ss_same)] * s_id, axis=0) + np.stack(
            [np.diag(self.ss_same)] * s_id, axis=1
        )
        self.ss_diff -= 2 * self.ss_same
        self.top_aid = self.voter.fusion_for_case(self.ans_set, inv_ans_id, 1)

    def softmax(self, input, axis=None):
        return np.exp(input) / np.exp(input).sum(axis)

    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def update_with_scgraph(self):
        prefer = self.softmax(self.ans_emb @ self.src_emb.T, axis=0)
        sample_src, sample_ans = random.choice(self.scgraph)
        grad = prefer[:, [sample_src]] @ self.src_emb[[sample_src]]
        pos_grad = grad - self.src_emb[sample_src]
        self.ans_emb[sample_ans] -= self.lr * pos_grad[sample_ans]
        self.ans_emb[~self.sa_mask[sample_src]] -= (
            self.lr * grad[~self.sa_mask[sample_src]]
        )
        grad_for_src = (prefer[:, sample_src] @ self.ans_emb) / (
            prefer[sample_ans, sample_src] * len(self.ans_set)
        )
        self.src_emb[sample_src] -= self.lr * (grad_for_src - self.ans_emb[sample_ans])

    def update_with_ssgraph(self):
        s1 = np.random.randint(0, self.source_num)
        s2 = np.random.randint(0, self.source_num)
        param1 = (
            2 - self.ss_diff[s1, s2] - self.ss_same[s1, s2] - self.alpha - self.beta
        )
        param1 *= 1.0 - self.sigmoid(self.src_emb[s1] @ self.src_emb[s2])
        param2 = self.ss_diff[s1, s2] + self.beta - 1
        grad_for_src = param1 * self.src_emb[s2] + param2 * self.src_emb[s2]
        self.src_emb[s1] -= self.lr * grad_for_src

    def loss_fn(self):
        sc_loss = (
            self.softmax(self.ans_emb @ self.src_emb.T, axis=0) * self.sa_mask.T
        ).sum()
        joint_prob = self.sigmoid(self.src_emb @ self.src_emb.T)
        cond_prob = np.power(joint_prob, self.ss_same) * np.power(
            1.0 - joint_prob, self.ss_diff
        )
        prior_prob = np.power(joint_prob, self.alpha - 1.0) * np.power(
            1.0 - joint_prob, self.beta - 1.0
        )
        prior_prob /= prior_prob.sum()
        ss_loss = (cond_prob + prior_prob).sum()
        return -(sc_loss + ss_loss)

    def iterate_fusion(self, **kwargs):
        its = 0
        while True:
            its += 1
            loss_prev = self.loss_fn()
            self.update_with_scgraph()
            self.update_with_ssgraph()
            loss_now = self.loss_fn()
            if abs(loss_now - loss_prev) <= self.converge_rate or its == self.max_iters:
                break
        truth_emb = self.ans_emb[self.top_aid].mean(axis=0)
        norm_a = np.linalg.norm(truth_emb)
        norm_b = np.linalg.norm(self.ans_emb, axis=-1)
        self.veracity = (self.ans_emb @ truth_emb) / (norm_b * norm_a)
        return kwargs["threshold"]


class LTMFusion:
    def __init__(self, source_num, alpha_0, alpha_1, beta, max_iters, burnin, thin):
        self.source_num = source_num
        self.max_iters = max_iters
        self.alpha = np.stack([np.array(alpha_0), np.array(alpha_1)], axis=0)
        self.beta = np.array(beta)
        self.burnin = burnin
        self.thin = thin

    def prepare_for_fusion(self, cand_answer):
        self.ans_set = []
        a_id, s_id = 0, 0
        self.facts = {}
        src_w_ans = {}
        for src, pairs in cand_answer.items():
            for ans, _ in pairs:
                if ans not in self.facts:
                    self.facts[ans] = a_id
                    self.ans_set.append(ans)
                    a_id += 1
                if self.facts[ans] not in src_w_ans:
                    src_w_ans[self.facts[ans]] = [s_id]
                else:
                    src_w_ans[self.facts[ans]].append(s_id)
            if len(pairs) > 0:
                s_id += 1
        self.claims = {aid: [] for aid in range(a_id)}
        for ans, aid in self.facts.items():
            for sid in range(s_id):
                if sid in src_w_ans[aid]:
                    self.claims[aid].append((sid, True))
                else:
                    self.claims[aid].append((sid, False))

        self.true_size = np.zeros((a_id, s_id, 2), dtype=int)
        self.false_size = np.zeros((a_id, s_id, 2), dtype=int)
        self.fact_truth = np.zeros(a_id, dtype=int)
        self.veracity = np.zeros(a_id, dtype=float)
        for aid in self.facts.values():
            if np.random.rand() < 0.5:
                for sid, flag in self.claims[aid]:
                    self.false_size[aid, sid, int(flag)] += 1
            else:
                self.fact_truth[aid] = 1
                for sid, flag in self.claims[aid]:
                    self.true_size[aid, sid, int(flag)] += 1

    def gibbs_sample(self, expect=False, sample_size=None):
        for aid, claim in self.claims.items():
            is_truth = self.fact_truth[aid]
            pos_count = self.beta[is_truth]
            neg_count = self.beta[1 - is_truth]
            if is_truth == 1:
                pos_side = self.true_size
                neg_side = self.false_size
            else:
                pos_side = self.false_size
                neg_side = self.true_size
            for sid, flag in claim:
                pos_alpha = (
                    pos_side[aid, sid, int(flag)] + self.alpha[is_truth, int(flag)] - 1
                )
                pos_denom = pos_side[aid, sid].sum() - 1 + self.alpha[is_truth].sum()
                neg_alpha = (
                    neg_side[aid, sid, int(flag)] + self.alpha[1 - is_truth, int(flag)]
                )
                neg_denom = neg_side[aid, sid].sum() + self.alpha[1 - is_truth].sum()
                pos_count = pos_count * (pos_alpha / pos_denom)
                neg_count = neg_count * (neg_alpha / neg_denom)
            if np.random.rand() < neg_count / (pos_count + neg_count):
                self.fact_truth[aid] = 1 - self.fact_truth[aid]
                for sid, flag in claim:
                    if self.fact_truth[aid] == 1:
                        self.false_size[aid, sid, int(flag)] -= 1
                        self.true_size[aid, sid, int(flag)] += 1
                    else:
                        self.true_size[aid, sid, int(flag)] -= 1
                        self.false_size[aid, sid, int(flag)] += 1

            if expect:
                self.veracity[aid] += self.fact_truth[aid] / sample_size

    def iterate_fusion(self, **kwargs):
        for its in range(1, self.max_iters + 1):
            if (its > self.burnin) and (its % self.thin == 0):
                sample_size = (self.max_iters - self.burnin) / self.thin
                self.gibbs_sample(True, sample_size)
            else:
                self.gibbs_sample()
        return kwargs["threshold"]
