import numpy as np
import warnings

warnings.filterwarnings("ignore")


class EMFusioner:
    his_data_size = None

    def __init__(
        self,
        source_num,
        max_iters,
        theta,
        init_trust=0.90,
        history_size=50,
        temperature=0.5,
        usemeta=False,
    ):
        self.source_num = source_num
        self.theta = theta
        self.iters = max_iters
        self.usemeta = usemeta
        self.eps = 1e-5
        self.tau = temperature
        self.src_trust_his = np.full(self.source_num, init_trust, dtype=float)
        EMFusioner.his_data_size = np.full(
            (self.source_num, 1), history_size, dtype=float
        )

    def prepare_for_fusion(self, cand_answer):
        a_id = 0
        self.ans_set = []
        prior_prob = []
        src_ans_dict = {}
        src_w_ans = {}
        for src, pairs in cand_answer.items():
            for ans, match_score in pairs:
                self.ans_set.append(ans)
                prior_prob.append(match_score)
                if src in src_ans_dict:
                    src_ans_dict[src].append(a_id)
                else:
                    src_ans_dict[src] = [a_id]
                if ans in src_w_ans:
                    src_w_ans[ans].append(src)
                else:
                    src_w_ans[ans] = [src]
                a_id += 1
        self.veracity = np.array(prior_prob)
        self.sa_mask = np.full((self.source_num, a_id), False)
        self.as_mask = np.full((self.source_num, a_id), False)
        for src, ans_lst in src_ans_dict.items():
            self.sa_mask[src, ans_lst] = True
        for aid, ans in enumerate(self.ans_set):
            src_lst = src_w_ans[ans]
            self.as_mask[src_lst, aid] = True

    def calculate_history_component(self, cur_data_size):
        cur_data_size = cur_data_size + EMFusioner.his_data_size
        ratio = EMFusioner.his_data_size / cur_data_size
        history_component = ratio * self.src_trust_his[:, np.newaxis]
        return history_component, cur_data_size

    def gumbel_softmax(self, prob, axis, temperature, weight=1.0):
        prob = np.clip(prob, 1e-5, 1.0 - 1e-5)
        return np.exp(weight * -np.log(1.0 - prob) / temperature) / np.exp(
            weight * -np.log(1.0 - prob) / temperature
        ).sum(axis, keepdims=True)

    def update_trustworthy(self, history_comp, cur_data_size):
        ## obtain set O_tau: {o_tau|Pr(o_tau) >= Pr(o)}
        self.src_prob = np.stack([self.veracity] * self.source_num, axis=0)
        o_tau = self.src_prob[:, np.newaxis, :] >= self.veracity[:, np.newaxis]
        o_tau = np.transpose(o_tau, (1, 0, 2)) & self.sa_mask
        ## update Pr(d|o)
        self.src_prob = np.where(o_tau, self.veracity, 0.0)
        self.src_prob = np.sum(self.src_prob, axis=-1).T
        self.src_prob = history_comp + self.src_prob / cur_data_size
        ## update Pr(d^t)
        vote = self.as_mask.sum(axis=0)
        weighted = self.gumbel_softmax(
            self.veracity, axis=None, temperature=self.tau, weight=vote
        )
        self.src_trust = (self.src_prob * weighted).sum(axis=1)

    def update_veracity(self):
        ## normalize Pr(d^t|o)
        src_w_ans = np.where(
            self.as_mask.any(axis=1, keepdims=True),
            self.src_prob,
            np.zeros_like(self.src_prob),
        )
        src_prob_norm = self.gumbel_softmax(src_w_ans, axis=0, temperature=self.tau)
        ## update Pr(o|d)
        o_d_prob = np.where(
            self.as_mask,
            self.src_trust[:, np.newaxis],
            1.0 - self.src_trust[:, np.newaxis],
        )
        ## update Pr(o)
        self.veracity = src_prob_norm * np.log(
            o_d_prob * self.src_trust[:, np.newaxis] / self.src_prob
        )
        self.veracity = np.exp(self.veracity.sum(axis=0))

    def update_threshold(self, threshold, cur_data_size):
        ## obtain set O_tau: {o_tau|Pr(o_tau) >= Pr(o)}
        self.src_prob = np.stack([self.veracity] * self.source_num, axis=0)
        o_tau = self.src_prob[:, np.newaxis, :] >= self.veracity[:, np.newaxis]
        o_tau = np.transpose(o_tau, (1, 0, 2)) & self.sa_mask
        o_tau_size = o_tau.sum(axis=-1)
        ## gradient of Pr(D)
        size = o_tau_size.T / cur_data_size
        grad = self.sa_mask.sum(axis=-1) + (self.veracity * size).sum(axis=-1)
        ## threshold update
        sign = np.sign(self.src_trust - self.src_trust_his)
        threshold = threshold - self.theta * sign * grad
        self.src_trust_his = self.src_trust
        EMFusioner.his_data_size += self.sa_mask.sum(axis=1, keepdims=True)
        return np.clip(threshold, 0.5, 0.99)

    def iterate_fusion(self, **kwargs):
        his_comp, cur_size = self.calculate_history_component(
            self.sa_mask.sum(axis=1, keepdims=True)
        )
        self.src_prob = np.stack([self.veracity] * self.source_num, axis=0)
        for its in range(self.iters):
            self.update_trustworthy(his_comp, cur_size)
            self.update_veracity()
        thres = self.update_threshold(kwargs["threshold"], cur_size)
        return thres
