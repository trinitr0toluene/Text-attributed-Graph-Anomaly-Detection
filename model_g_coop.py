import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import model
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch_geometric.nn.inits import glorot
import copy

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb


class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p


class FeatureLoss(nn.Module):
    def __init__(self, ):
        super(FeatureLoss, self).__init__()

    def forward(self, text_features, fix_label_features, label):

        loss_num = 0
        loss = torch.tensor(0.).to(text_features.device)

        for i in range(label.size(0)):
            if torch.sum(label != label[i]) > 0:
                index = label[i]
                dist_ap = (1 - F.cosine_similarity(text_features[index].unsqueeze(0),
                                                   fix_label_features[index].unsqueeze(0).detach()).squeeze())
                loss_num += 1
                loss += dist_ap

        if loss_num != 0:
            return loss / loss_num
        else:
            return loss


class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, g_texts):
        super().__init__()
        self.vars = nn.ParameterList()
        self.shots = args.k_spt
        n_cls = len(classnames)
        n_ctx = args.coop_n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]



        # random initialization
        if args.ctx_init:
            # use given words to initialize context vectors
            if args.class_specific:
                ctx_vectors = []
                for ctx_list in g_texts:
                    prompt = model.tokenize(ctx_list, context_length=args.context_length)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompt).type(dtype)
                    ctx_vector = embedding[:, 1: 1 + n_ctx, :]
                    ctx_vector = torch.mean(ctx_vector, dim=0)
                    ctx_vectors.append(ctx_vector)
                ctx_vectors = torch.stack(ctx_vectors)
            else:
                temp = []
                for ctx_list in g_texts:
                    temp += ctx_list
                prompt = model.tokenize(temp, context_length=args.context_length)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vector = embedding[:, 1: 1 + n_ctx, :]
                ctx_vectors = torch.mean(ctx_vector, dim=0)
            # print('ctx_vectors.shape', ctx_vectors.shape)
        else:
            if args.class_specific:
                # print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                # print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.vars.append(self.ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.name_lens = name_lens
        self.min_len = min(self.name_lens)  # 1
        if self.min_len > 1:
            print("origin len is ", name_lens)
            classnames = self.revise_classnames(classnames, name_lens, self.min_len)
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            print("later len is ", name_lens)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [model.tokenize(p, context_length=args.context_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_suffix_test", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.position

        self._init_suffix_dict(classnames, clip_model, dtype)
        self._get_token_classes(dtype)

    def revise_classnames(self, classnames, name_lens, min_len):
        if min(name_lens) < min_len:
            for i in range(len(classnames)):
                if name_lens[i] < min_len:
                    classnames[i] = ("<|startoftext|> "*(min_len - name_lens[i])) + classnames[i]
        return classnames

    def _init_suffix_dict(self, classnames, clip_model, dtype):

        self.suffix_classes = {}
        for name in classnames:
            self.suffix_classes[name] = clip_model.token_embedding(model.tokenize(name)).type(dtype)

    def _get_token_classes(self, dtype):

        if self.training:
            self.token_classes_all = torch.cat([self.suffix_classes[name] for name in self.suffix_classes]).type(dtype)
            self.token_classes = self.token_classes_all[:, 1:self.min_len + 1, :]
            if 1:
                nn.init.normal_(self.token_classes, std=0.02)
            self.token_classes = nn.Parameter(self.token_classes)
            self.fix_token = copy.deepcopy(self.token_classes)
            self.fix_token.requires_grad = False
        else:
            pass

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def parameters(self):
        return self.vars


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model, g_texts):
        super().__init__()
        self.args = args
        self.prompt_learner = PromptLearner(args, classnames, clip_model, g_texts)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.gnn
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.loss_feature_fn = FeatureLoss()
        self.shots = args.k_spt

    def _get_origin_feature(self):
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner.forward()
        fix_label_features = self.text_encoder(prompts, tokenized_prompts)
        return fix_label_features

    def distillation(self, t, s, T=2):
        p = F.softmax(t / T, dim=1)
        loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
        return loss

    def forward(self, s_n, x, adj, label=None):
        image_features = self.image_encoder(x, adj)[s_n]
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            fix_label_features = self._get_origin_feature()
            fix_label_features = fix_label_features / fix_label_features.norm(dim=-1, keepdim=True)

            fix_logists = logit_scale * image_features @ fix_label_features.t()
            loss_logits = self.distillation(logits, fix_logists)

            loss_feature = self.loss_feature_fn(text_features, fix_label_features, label)
            loss_paramter = F.mse_loss(self.prompt_learner.token_classes, self.prompt_learner.fix_token.detach(),
                                       reduction="sum")

            entorpy_loss = F.cross_entropy(logits, label)
            return entorpy_loss + 1 / self.shots * loss_paramter + loss_feature + 0.05 * loss_logits

        return logits


class CoOp(nn.Module):
    """Context Optimization (CoOp).
    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def __init__(self, args, classnames, clip_model, g_texts, device, is_plus=False):
        super().__init__()
        self.args = args
        self.classnames = classnames
        self.model = CustomCLIP(args, classnames, clip_model, g_texts)
        self.answering = torch.nn.Sequential(torch.nn.Linear(args.gnn_output, args.n_way),
                                             torch.nn.Softmax(dim=1)).to(device)
        if is_plus:
            self.graph_prompt = GPF_plus(args.gnn_hid, args.pnum).to(device)
        else:
            self.graph_prompt = GPF(args.gnn_hid).to(device)

        # print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, args.OPTIM)
        self.model.to(device)
        # model_prompt_group = [{'params': self.graph_prompt.parameters()},
        #                       {'params': self.model.prompt_learner.parameters()},
        # #                       {'params': self.answering.parameters()}]
        # self.optim = optim.Adam(model_prompt_group, lr=args.prompt_lr)

        self.optim = optim.Adam(self.model.prompt_learner.parameters())

    def forward(self, s_n, x, adj, label, training=True):
        if training:
            loss = self.model(s_n, x, adj, label)
            self.optim.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            self.optim.step()
        else:
            logits = self.model(s_n, x, adj, label)
            return logits

    # def forward(self, s_n, batch, label, training=True):
    #     if training:
    #         loss = self.model(s_n, batch, label)
    #         self.optim.zero_grad()
    #         torch.cuda.empty_cache()
    #         loss.backward()
    #         self.optim.step()
    #     else:
    #         logits = self.model(s_n, batch, label)
    #         return logits
