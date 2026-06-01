from __future__ import annotations

import random


class Augmenter:
    """Data augmentation operators from original Ditto."""

    def augment(self, tokens, labels, op="del"):
        if "del" in op:
            span_len = random.randint(1, 2)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            new_tokens = tokens[:pos1] + tokens[pos2 + 1 :]
            new_labels = labels[:pos1] + labels[pos2 + 1 :]
        elif "swap" in op:
            span_len = random.randint(2, 4)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            sub_arr = tokens[pos1 : pos2 + 1]
            random.shuffle(sub_arr)
            new_tokens = tokens[:pos1] + sub_arr + tokens[pos2 + 1 :]
            new_labels = labels[:pos1] + ["O"] * (pos2 - pos1 + 1) + labels[pos2 + 1 :]
        elif "drop_len" in op:
            all_lens = [len(token) for token, label in zip(tokens, labels) if label == "O"]
            if not all_lens:
                return tokens, labels
            target_lens = random.choices(all_lens, k=1)
            new_tokens, new_labels = [], []
            for token, label in zip(tokens, labels):
                if label != "O" or len(token) not in target_lens:
                    new_tokens.append(token)
                    new_labels.append(label)
            return new_tokens, new_labels
        elif "drop_sym" in op:
            def drop_sym(token):
                return "".join([ch if ch.isalnum() else " " for ch in token]).strip()

            dropped_tokens = [drop_sym(token) for token in tokens]
            new_tokens, new_labels = [], []
            for token, d_token, label in zip(tokens, dropped_tokens, labels):
                if random.randint(0, 4) != 0 or label != "O":
                    new_tokens.append(token)
                    new_labels.append(label)
                elif d_token:
                    new_tokens.append(d_token)
                    new_labels.append(label)
            return new_tokens, new_labels
        elif "drop_same" in op:
            left_token, right_token = set(), set()
            left = True
            for token, label in zip(tokens, labels):
                if label == "O":
                    token = token.lower()
                    if left:
                        left_token.add(token)
                    else:
                        right_token.add(token)
                if token == "[SEP]":
                    left = False

            same = left_token & right_token
            if not same:
                return tokens, labels
            targets = random.choices(list(same), k=1)
            new_tokens, new_labels = [], []
            for token, label in zip(tokens, labels):
                if token.lower() not in targets or label != "O":
                    new_tokens.append(token)
                    new_labels.append(label)
            return new_tokens, new_labels
        elif "drop_token" in op:
            new_tokens, new_labels = [], []
            for token, label in zip(tokens, labels):
                if label != "O" or random.randint(0, 4) != 0:
                    new_tokens.append(token)
                    new_labels.append(label)
            return new_tokens, new_labels
        elif "ins" in op:
            pos = self.sample_position(tokens, labels)
            if pos < 0:
                return tokens, labels
            symbol = random.choice("-*.,#&")
            new_tokens = tokens[:pos] + [symbol] + tokens[pos:]
            new_labels = labels[:pos] + ["O"] + labels[pos:]
            return new_tokens, new_labels
        elif "append_col" in op:
            col_starts = [i for i in range(len(tokens)) if tokens[i] == "COL"]
            col_ends = [0] * len(col_starts)
            col_lens = [0] * len(col_starts)
            for i, pos in enumerate(col_starts):
                if i == len(col_starts) - 1:
                    col_lens[i] = len(tokens) - pos
                    col_ends[i] = len(tokens) - 1
                else:
                    col_lens[i] = col_starts[i + 1] - pos
                    col_ends[i] = col_starts[i + 1] - 1
                if tokens[col_ends[i]] == "[SEP]":
                    col_ends[i] -= 1
                    col_lens[i] -= 1
                    break

            candidates = [i for i, le in enumerate(col_lens) if le > 0]
            if len(candidates) >= 2:
                idx1, idx2 = random.sample(candidates, k=2)
                start1, end1 = col_starts[idx1], col_ends[idx1]
                sub_tokens = tokens[start1 : end1 + 1]
                sub_labels = labels[start1 : end1 + 1]
                val_pos = 0
                for i, token in enumerate(sub_tokens):
                    if token == "VAL":
                        val_pos = i + 1
                        break
                sub_tokens = sub_tokens[val_pos:]
                sub_labels = sub_labels[val_pos:]

                end2 = col_ends[idx2]
                out_t, out_l = [], []
                for i in range(len(tokens)):
                    if start1 <= i <= end1:
                        continue
                    out_t.append(tokens[i])
                    out_l.append(labels[i])
                    if i == end2:
                        out_t += sub_tokens
                        out_l += sub_labels
                return out_t, out_l
            return tokens, labels
        elif "drop_col" in op:
            col_starts = [i for i in range(len(tokens)) if tokens[i] == "COL"]
            col_ends = [0] * len(col_starts)
            col_lens = [0] * len(col_starts)
            for i, pos in enumerate(col_starts):
                if i == len(col_starts) - 1:
                    col_lens[i] = len(tokens) - pos
                    col_ends[i] = len(tokens) - 1
                else:
                    col_lens[i] = col_starts[i + 1] - pos
                    col_ends[i] = col_starts[i + 1] - 1
                if tokens[col_ends[i]] == "[SEP]":
                    col_ends[i] -= 1
                    col_lens[i] -= 1
            candidates = [i for i, le in enumerate(col_lens) if le <= 8]
            if candidates:
                idx = random.choice(candidates)
                start, end = col_starts[idx], col_ends[idx]
                new_tokens = tokens[:start] + tokens[end + 1 :]
                new_labels = labels[:start] + labels[end + 1 :]
                return new_tokens, new_labels
            return tokens, labels

        return new_tokens, new_labels

    def augment_sent(self, text, op="all"):
        if " [SEP] " in text and random.randint(0, 1) == 0:
            left, right = text.split(" [SEP] ", 1)
            text = right + " [SEP] " + left

        tokens = text.split(" ")
        labels = []
        for token in tokens:
            if token in ["COL", "VAL"]:
                labels.append("HD")
            elif token in ["[CLS]", "[SEP]"]:
                labels.append("<SEP>")
            else:
                labels.append("O")

        if op == "all":
            N = 3
            ops = ["del", "swap", "drop_col", "append_col"]
            for chosen in random.choices(ops, k=N):
                tokens, labels = self.augment(tokens, labels, op=chosen)
        else:
            tokens, labels = self.augment(tokens, labels, op=op)

        return " ".join(tokens)

    @staticmethod
    def sample_span(tokens, labels, span_len=3):
        candidates = []
        for idx in range(len(tokens)):
            if idx + span_len - 1 < len(labels) and "".join(labels[idx : idx + span_len]) == "O" * span_len:
                candidates.append((idx, idx + span_len - 1))
        if not candidates:
            return -1, -1
        return random.choice(candidates)

    @staticmethod
    def sample_position(tokens, labels):
        candidates = [idx for idx in range(len(tokens)) if labels[idx] == "O"]
        if not candidates:
            return -1
        return random.choice(candidates)
