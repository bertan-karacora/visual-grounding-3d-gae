import logging

import torch

import project.models.attention as attention
from project.models.attention import SelfAttention, CrossAttention, SelfAttentionSpatial
import project.models.bert as bert
from project.models.bert import EncoderLanguageBERT
from project.models.gae import EncoderGNN
from project.models.encoder_position import EncoderPositionMLP
from project.models.pointnetpp import EncoderPointsPointNetPP
from project.models.mlp import MLP


_LOGGER = logging.getLogger(__name__)


class ReferralLayer(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_heads, prob_dropout=0.1, use_cross_attention=True):
        super().__init__()

        self.mlp = None
        self.module_attention_cross = None
        self.module_attention_self = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout
        self.use_cross_attention = use_cross_attention

        self._init()

    def _init(self):
        self.module_attention_self = SelfAttention(self.num_channels_in, self.num_heads, prob_dropout=self.prob_dropout, use_batch_first=True)
        if self.use_cross_attention:
            self.module_attention_cross = CrossAttention(self.num_channels_in, self.num_heads, prob_dropout=self.prob_dropout, use_batch_first=True)

        self.mlp = MLP(num_channels_in=self.num_channels_in, nums_channels_hidden=self.nums_channels_hidden, num_channels_out=self.num_channels_in, name_layer_norm="LayerNorm", prob_dropout=self.prob_dropout)

    def forward(self, input):
        output = input["embeddings_object"]

        if self.module_attention_cross is not None:
            output = self.module_attention_cross(output, input["embeddings_language"], mask_padding_key=input["mask_language"])
            output = output * input["mask_object"][..., None]

        output = self.module_attention_self(output, mask_padding_key=input["mask_object"])
        output = output * input["mask_object"][..., None]

        output = self.mlp(output)
        output = torch.nn.functional.layer_norm(output, (self.num_channels_in,))

        return output


class Referral(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_layers=4, num_heads=12, prob_dropout=0.1, use_cross_attention=True, use_encoding_positions=True):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prob_dropout = prob_dropout
        self.use_cross_attention = use_cross_attention
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.layers = torch.nn.ModuleList(
            [
                ReferralLayer(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=self.nums_channels_hidden,
                    num_heads=self.num_heads,
                    prob_dropout=self.prob_dropout,
                    use_cross_attention=self.use_cross_attention,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.apply(bert._init_weights)

    def forward(self, input):
        if self.use_encoding_positions:
            positions = torch.arange(input["embeddings_language"].shape[1], dtype=torch.float32, device=input["embeddings_language"].device)[:, None]
            encoding_position = attention.encode_position_1d(positions, shape=input["embeddings_language"].shape[1:], device=input["embeddings_language"].device)
            input["embeddings_language"] += encoding_position

            encoding_position = attention.encode_position_3d(input["positions"], input["embeddings_object"].shape[-1], device=input["embeddings_object"].device)
            input["embeddings_object"] += encoding_position

        output = input["embeddings_object"]
        for layer in self.layers:
            input_layer = dict(
                embeddings_object=output,
                mask_object=input["mask_object"],
                embeddings_language=input["embeddings_language"],
                mask_language=input["mask_language"],
            )
            output = layer(input_layer)

        return output


class ReferralSelfAttention(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_layers=4, num_heads=12, prob_dropout=0.1, use_encoding_positions=True):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prob_dropout = prob_dropout
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.layers = torch.nn.ModuleList(
            [
                ReferralLayer(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=self.nums_channels_hidden,
                    num_heads=self.num_heads,
                    prob_dropout=self.prob_dropout,
                    use_cross_attention=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.apply(bert._init_weights)

    def forward(self, input):
        if self.use_encoding_positions:
            positions = torch.arange(input["embeddings_language"].shape[1], dtype=torch.float32, device=input["embeddings_language"].device)[:, None]
            encoding_position = attention.encode_position_1d(positions, shape=input["embeddings_language"].shape[1:], device=input["embeddings_language"].device)
            input["embeddings_language"] += encoding_position

            encoding_position = attention.encode_position_3d(input["positions"], input["embeddings_object"].shape[-1], device=input["embeddings_object"].device)
            input["embeddings_object"] += encoding_position

        output = torch.cat((input["embeddings_object"], input["embeddings_language"]), dim=1)
        mask = torch.cat((input["mask_object"], input["mask_language"]), dim=1)

        for layer in self.layers:
            input_layer = dict(embeddings_object=output, mask_object=mask)
            output = layer(input_layer)

        output = torch.split(output, [input["embeddings_object"].shape[1], input["embeddings_language"].shape[1]], dim=1)[0]

        return output


class ReferralLayerSpatial(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_heads, prob_dropout=0.1):
        super().__init__()

        self.mlp = None
        self.module_attention_self = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        self.module_attention_self = SelfAttentionSpatial(self.num_channels_in, self.num_heads, prob_dropout=self.prob_dropout, use_batch_first=True)
        self.mlp = MLP(num_channels_in=self.num_channels_in, nums_channels_hidden=self.nums_channels_hidden, num_channels_out=self.num_channels_in, name_layer_norm="LayerNorm", prob_dropout=self.prob_dropout)

    def forward(self, input):
        output = input["embeddings_object"]

        output = self.module_attention_self(output, input["pairwise_locs"], mask_padding_key=input["mask_object"])
        output = output * input["mask_object"][..., None]

        output = self.mlp(output)
        output = torch.nn.functional.layer_norm(output, (self.num_channels_in,))

        return output


def init_weights_3DVisTA(module):
    """Initialize the weights"""
    if isinstance(module, torch.nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ReferralSpatial(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_layers=4, num_heads=12, prob_dropout=0.1, use_encoding_positions=True):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prob_dropout = prob_dropout
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.layers = torch.nn.ModuleList(
            [
                ReferralLayerSpatial(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=self.nums_channels_hidden,
                    num_heads=self.num_heads,
                    prob_dropout=self.prob_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.apply(init_weights_3DVisTA)

    def forward(self, input):
        if self.use_encoding_positions:
            encoding_position = attention.encode_position_3d(input["pos"], input["embeddings_object"].shape[-1], device=input["embeddings_object"].device)
            input["embeddings_object"] += encoding_position

        pairwise_locs = input["pos"].unsqueeze(2) - input["pos"].unsqueeze(1)

        # pairwise distances
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, dim=3) + 1e-10)

        # normalize distances
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / max_dists.view(-1, 1, 1)

        # 2D pairwise distances (using first two coords)
        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, dim=3) + 1e-10)

        # final stack
        pairwise_locs = torch.stack(
            [
                norm_pairwise_dists,
                pairwise_locs[..., 2] / pairwise_dists,
                pairwise_dists_2d / pairwise_dists,
                pairwise_locs[..., 1] / pairwise_dists_2d,
                pairwise_locs[..., 0] / pairwise_dists_2d,
            ],
            dim=3,
        )

        output = input["embeddings_object"]
        for layer in self.layers:
            input_layer = dict(
                embeddings_object=output,
                mask_object=input["mask_object"],
                pairwise_locs=pairwise_locs,
            )
            output = layer(input_layer)

        return output


class HeadGroundingMLP(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, prob_dropout=None):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        self.mlp = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=self.nums_channels_hidden,
            num_channels_out=1,
            name_layer_norm="LayerNorm",
            kwargs_norm=dict(eps=1e-12),
            prob_dropout=self.prob_dropout,
        )

    def forward(self, input):
        output = input["embeddings"]
        output = self.mlp(output)
        output = output[..., 0]
        output = output.masked_fill_(~input["mask"], -float("inf"))

        return output


class ModelVisualGroundingGNN(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_hidden,
        nums_channels_hidden_encoder_position,
        num_channels_out_encoder_position,
        nums_channels_hidden_projector_class,
        num_channels_out_projector_class,
        nums_channels_hidden_encoder_gnn,
        nums_channels_hidden_projector_language,
        nums_channels_hidden_referral,
        nums_channels_hidden_head,
        num_samples,
        num_layers_referral,
        name_layer_conv="GCNConv",
        mode_norm="PN-SCS",
        scale_norm=20,
        prob_dropout_encoder_points=None,
        prob_dropout_projector_class=None,
        prob_dropout_encoder_gnn=None,
        prob_dropout_head=None,
        prob_dropout_projector_language=None,
        prob_dropout_referral=None,
        prob_dropout_encoder_position=None,
        use_residual=False,
        use_concat=True,
        use_norm_pair=False,
        mode_jk="max",
        use_encoding_positions=True,
    ):
        super().__init__()

        self.referral = None
        self.encoder_language = None
        self.encoder_points = None
        self.encoder_position = None
        self.head_grounding = None
        self.mode_jk = mode_jk
        self.mode_norm = mode_norm
        self.name_layer_conv = name_layer_conv
        self.num_channels_in = num_channels_in
        self.num_channels_hidden = num_channels_hidden
        self.nums_channels_hidden_projector_class = nums_channels_hidden_projector_class
        self.nums_channels_hidden_encoder_gnn = nums_channels_hidden_encoder_gnn
        self.nums_channels_hidden_encoder_position = nums_channels_hidden_encoder_position
        self.nums_channels_hidden_head = nums_channels_hidden_head
        self.nums_channels_hidden_projector_language = nums_channels_hidden_projector_language
        self.nums_channels_hidden_referral = nums_channels_hidden_referral
        self.num_channels_out_projector_class = num_channels_out_projector_class
        self.num_channels_out_encoder_position = num_channels_out_encoder_position
        self.num_layers_referral = num_layers_referral
        self.num_samples = num_samples
        self.prob_dropout_projector_class = prob_dropout_projector_class
        self.prob_dropout_encoder_gnn = prob_dropout_encoder_gnn
        self.prob_dropout_encoder_points = prob_dropout_encoder_points
        self.prob_dropout_head = prob_dropout_head
        self.prob_dropout_projector_language = prob_dropout_projector_language
        self.prob_dropout_encoder_position = prob_dropout_encoder_position
        self.prob_dropout_referral = prob_dropout_referral
        self.projector_language = None
        self.scale_norm = scale_norm
        self.use_concat = use_concat
        self.use_norm_pair = use_norm_pair
        self.use_residual = use_residual
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.encoder_points = EncoderPointsPointNetPP(prob_dropout=self.prob_dropout_encoder_points)
        self.encoder_position = EncoderPositionMLP(
            num_channels_in=6,
            nums_channels_hidden=self.nums_channels_hidden_encoder_position,
            num_channels_out=self.num_channels_out_encoder_position,
            prob_dropout=self.prob_dropout_encoder_position,
        )
        self.projector_class = MLP(
            num_channels_in=300,
            nums_channels_hidden=self.nums_channels_hidden_projector_class,
            num_channels_out=self.num_channels_out_projector_class,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_projector_class,
        )
        self.encoder_fusion = MLP(
            num_channels_in=768 + self.num_channels_out_encoder_position + self.num_channels_out_projector_class,
            nums_channels_hidden=(self.num_channels_hidden, self.num_channels_hidden),
            num_channels_out=self.num_channels_hidden,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_encoder_position,
        )
        self.encoder_gnn = EncoderGNN(
            num_channels_in=self.num_channels_out_projector_class,
            nums_channels_hidden=self.nums_channels_hidden_encoder_gnn,
            num_channels_out=self.num_channels_hidden,
            name_layer_conv=self.name_layer_conv,
            mode_norm=self.mode_norm,
            scale_norm=self.scale_norm,
            prob_dropout=self.prob_dropout_encoder_gnn,
            use_residual=self.use_residual,
            mode_jk=self.mode_jk,
            use_norm_pair=self.use_norm_pair,
        )

        self.encoder_language = EncoderLanguageBERT(name_weights="bert-base-uncased", num_channels_out=768, num_layers=4, num_heads=12, type_vocab_size=2)
        self.projector_language = MLP(
            num_channels_in=768,
            nums_channels_hidden=self.nums_channels_hidden_projector_language,
            num_channels_out=self.num_channels_hidden,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_projector_language,
        )

        if self.use_concat:
            self.referral = ReferralSelfAttention(
                num_channels_in=self.num_channels_hidden,
                nums_channels_hidden=self.nums_channels_hidden_referral,
                num_layers=self.num_layers_referral,
                prob_dropout=self.prob_dropout_referral,
                use_encoding_positions=self.use_encoding_positions,
            )
        else:
            self.referral = Referral(
                num_channels_in=self.num_channels_hidden,
                nums_channels_hidden=self.nums_channels_hidden_referral,
                num_layers=self.num_layers_referral,
                prob_dropout=self.prob_dropout_referral,
                use_encoding_positions=self.use_encoding_positions,
            )

        self.head_grounding = HeadGroundingMLP(num_channels_in=self.num_channels_hidden, nums_channels_hidden=self.nums_channels_hidden_head, prob_dropout=self.prob_dropout_head)

    def forward(self, input):
        input_points = input["features"][None, ...]
        output_features = self.encoder_points(input_points)
        output_features = output_features[0]

        input_position = torch.cat((input["pos"], input["size"]), dim=1)
        output_position = self.encoder_position(input_position)

        input_class = input["embeddings_class"]
        output_class = self.projector_class(input_class)
        output_class = torch.nn.functional.layer_norm(output_class, (self.num_channels_out_projector_class,))

        input_fusion = torch.cat((output_features, output_class, output_position), dim=1)
        output_fusion = self.encoder_fusion(input_fusion)
        output_fusion = torch.nn.functional.layer_norm(output_fusion, (self.num_channels_hidden,))

        input_gnn = dict(
            features=output_fusion,
            edge_index=input["edge_index"],
        )
        output_gnn = self.encoder_gnn(input_gnn)

        input_language = dict(
            sentence=input["sentence"],
            device=input["features"].device,
        )
        output_language = self.encoder_language(input_language)

        output_language["embeddings"] = self.projector_language(output_language["embeddings"])
        output_language["embeddings"] = torch.nn.functional.layer_norm(output_language["embeddings"], (self.num_channels_hidden,))

        # Have output_language of shape (B, MAX_NUM_TOKENS, C) and masks
        # Have output_gnn embeddings of shape (N, num_channels_HIDDEN) where N is the number of objects flattened over the batch and input["batch"] can be used for association

        # Regroup objects by batch
        list_embeddings_object = []
        list_positions = []
        for b in range(len(input["ptr"]) - 1):
            list_embeddings_object.append(output_gnn[input["batch"] == b])
            list_positions.append(input["pos"][input["batch"] == b])
        # Pad to [B, N_max, D]
        embeddings_object = torch.nn.utils.rnn.pad_sequence(list_embeddings_object, batch_first=True)
        positions = torch.nn.utils.rnn.pad_sequence(list_positions, batch_first=True)

        mask_object = torch.zeros(embeddings_object.size(0), embeddings_object.size(1), dtype=torch.bool, device=input["features"].device)
        for b, objs_b in enumerate(list_embeddings_object):
            mask_object[b, : objs_b.size(0)] = True

        input_referral = dict(
            embeddings_object=embeddings_object,
            mask_object=mask_object,
            embeddings_language=output_language["embeddings"],
            mask_language=output_language["mask"],
            positions=positions,
        )
        output_referral = self.referral(input_referral)

        input_grounding = dict(embeddings=output_referral, mask=mask_object)
        output_logits = self.head_grounding(input_grounding)

        output = dict(
            logits=output_logits,
            mask=mask_object,
        )

        return output


class ModelVisualGroundingSpatialTransformer(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_hidden,
        nums_channels_hidden_encoder_position,
        num_channels_out_encoder_position,
        nums_channels_hidden_projector_class,
        num_channels_out_projector_class,
        nums_channels_hidden_projector_language,
        nums_channels_hidden_referral,
        nums_channels_hidden_head,
        num_samples,
        num_layers_referral,
        name_layer_conv="GCNConv",
        mode_norm="PN-SCS",
        scale_norm=20,
        prob_dropout_encoder_points=None,
        prob_dropout_projector_class=None,
        prob_dropout_encoder_gnn=None,
        prob_dropout_head=None,
        prob_dropout_projector_language=None,
        prob_dropout_encoder_position=None,
        prob_dropout_referral=None,
        use_residual=False,
        use_concat=True,
        use_norm_pair=False,
        mode_jk="max",
        use_encoding_positions=True,
    ):
        super().__init__()

        self.referral = None
        self.encoder_language = None
        self.encoder_points = None
        self.encoder_position = None
        self.head_grounding = None
        self.mode_jk = mode_jk
        self.mode_norm = mode_norm
        self.name_layer_conv = name_layer_conv
        self.num_channels_in = num_channels_in
        self.num_channels_hidden = num_channels_hidden
        self.nums_channels_hidden_projector_class = nums_channels_hidden_projector_class
        self.nums_channels_hidden_encoder_position = nums_channels_hidden_encoder_position
        self.nums_channels_hidden_head = nums_channels_hidden_head
        self.nums_channels_hidden_projector_language = nums_channels_hidden_projector_language
        self.nums_channels_hidden_referral = nums_channels_hidden_referral
        self.num_channels_out_projector_class = num_channels_out_projector_class
        self.num_channels_out_encoder_position = num_channels_out_encoder_position
        self.num_layers_referral = num_layers_referral
        self.num_samples = num_samples
        self.prob_dropout_projector_class = prob_dropout_projector_class
        self.prob_dropout_encoder_gnn = prob_dropout_encoder_gnn
        self.prob_dropout_encoder_points = prob_dropout_encoder_points
        self.prob_dropout_encoder_position = prob_dropout_encoder_position
        self.prob_dropout_head = prob_dropout_head
        self.prob_dropout_projector_language = prob_dropout_projector_language
        self.prob_dropout_referral = prob_dropout_referral
        self.projector_language = None
        self.scale_norm = scale_norm
        self.use_concat = use_concat
        self.use_norm_pair = use_norm_pair
        self.use_residual = use_residual
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.encoder_points = EncoderPointsPointNetPP(prob_dropout=self.prob_dropout_encoder_points)
        self.encoder_position = EncoderPositionMLP(
            num_channels_in=6,
            nums_channels_hidden=self.nums_channels_hidden_encoder_position,
            num_channels_out=self.num_channels_out_encoder_position,
            prob_dropout=self.prob_dropout_encoder_position,
        )
        self.projector_class = MLP(
            num_channels_in=300,
            nums_channels_hidden=self.nums_channels_hidden_projector_class,
            num_channels_out=self.num_channels_out_projector_class,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_projector_class,
        )
        self.encoder_fusion = MLP(
            num_channels_in=768 + self.num_channels_out_encoder_position + self.num_channels_out_projector_class,
            nums_channels_hidden=(self.num_channels_hidden, self.num_channels_hidden),
            num_channels_out=self.num_channels_hidden,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_encoder_position,
        )
        self.encoder_language = EncoderLanguageBERT(name_weights="bert-base-uncased", num_channels_out=768, num_layers=4, num_heads=12, type_vocab_size=2)
        self.projector_language = MLP(
            num_channels_in=768,
            nums_channels_hidden=self.nums_channels_hidden_projector_language,
            num_channels_out=self.num_channels_hidden,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_projector_language,
        )

        self.transformer_spatial = TransformerSpatial(
            num_channels_in=self.num_channels_hidden, nums_channels_hidden=(self.num_channels_hidden,), num_layers=4, num_heads=12, prob_dropout=self.prob_dropout_referral, use_encoding_positions=self.use_encoding_positions
        )

        if self.use_concat:
            self.referral = ModelReferral(
                num_channels_in=self.num_channels_hidden,
                nums_channels_hidden=self.nums_channels_hidden_referral,
                num_layers=self.num_layers_referral,
                prob_dropout=self.prob_dropout_referral,
                use_encoding_positions=self.use_encoding_positions,
            )
        else:
            self.referral = ModelReferral2(
                num_channels_in=self.num_channels_hidden,
                nums_channels_hidden=self.nums_channels_hidden_referral,
                num_layers=self.num_layers_referral,
                prob_dropout=self.prob_dropout_referral,
                use_encoding_positions=self.use_encoding_positions,
            )

        self.head_grounding = HeadGroundingMLP(num_channels_in=self.num_channels_hidden, nums_channels_hidden=self.nums_channels_hidden_head, prob_dropout=self.prob_dropout_head)

    def forward(self, input):
        input_points = input["features"][None, ...]
        output_features = self.encoder_points(input_points)
        output_features = output_features[0]

        input_position = torch.cat((input["pos"], input["size"]), dim=1)
        output_position = self.encoder_position(input_position)

        input_class = input["embeddings_class"]
        output_class = self.projector_class(input_class)
        output_class = torch.nn.functional.layer_norm(output_class, (self.num_channels_out_projector_class,))

        input_fusion = torch.cat((output_features, output_class, output_position), dim=1)
        output_fusion = self.encoder_fusion(input_fusion)
        output_fusion = torch.nn.functional.layer_norm(output_fusion, (self.num_channels_hidden,))

        # Have output_language of shape (B, MAX_NUM_TOKENS, C) and masks
        # Have output_fusion embeddings of shape (N, num_channels_HIDDEN) where N is the number of objects flattened over the batch and input["batch"] can be used for association

        # Regroup objects by batch
        list_embeddings_object = []
        list_positions = []
        for b in range(len(input["ptr"]) - 1):
            list_embeddings_object.append(output_fusion[input["batch"] == b])
            list_positions.append(input["pos"][input["batch"] == b])
        # Pad to [B, N_max, D]
        embeddings_object = torch.nn.utils.rnn.pad_sequence(list_embeddings_object, batch_first=True)
        positions = torch.nn.utils.rnn.pad_sequence(list_positions, batch_first=True)

        mask_object = torch.zeros(embeddings_object.size(0), embeddings_object.size(1), dtype=torch.bool, device=input["features"].device)
        for b, objs_b in enumerate(list_embeddings_object):
            mask_object[b, : objs_b.size(0)] = True

        input_transformer_spatial = dict(
            embeddings_object=embeddings_object,
            mask_object=mask_object,
            pos=positions,
        )
        output_transformer_spatial = self.transformer_spatial(input_transformer_spatial)

        input_language = dict(
            sentence=input["sentence"],
            device=input["features"].device,
        )
        output_language = self.encoder_language(input_language)

        output_language["embeddings"] = self.projector_language(output_language["embeddings"])
        output_language["embeddings"] = torch.nn.functional.layer_norm(output_language["embeddings"], (self.num_channels_hidden,))

        input_referral = dict(
            embeddings_object=output_transformer_spatial,
            mask_object=mask_object,
            embeddings_language=output_language["embeddings"],
            mask_language=output_language["mask"],
            positions=positions,
        )
        output_referral = self.referral(input_referral)

        input_grounding = dict(embeddings=output_referral, mask=mask_object)
        output_logits = self.head_grounding(input_grounding)

        output = dict(
            logits=output_logits,
            mask=mask_object,
        )

        return output


class ModelVisualGrounding(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_hidden,
        nums_channels_hidden_encoder_position,
        num_channels_out_encoder_position,
        nums_channels_hidden_projector_class,
        num_channels_out_projector_class,
        nums_channels_hidden_projector_language,
        nums_channels_hidden_referral,
        nums_channels_hidden_head,
        num_samples,
        num_layers_referral,
        name_layer_conv="GCNConv",
        mode_norm="PN-SCS",
        scale_norm=20,
        prob_dropout_encoder_points=None,
        prob_dropout_projector_class=None,
        prob_dropout_encoder_gnn=None,
        prob_dropout_head=None,
        prob_dropout_projector_language=None,
        prob_dropout_encoder_position=None,
        prob_dropout_referral=None,
        use_residual=False,
        use_concat=True,
        use_norm_pair=False,
        mode_jk="max",
        use_encoding_positions=True,
    ):
        super().__init__()

        self.referral = None
        self.encoder_language = None
        self.encoder_points = None
        self.encoder_position = None
        self.head_grounding = None
        self.mode_jk = mode_jk
        self.mode_norm = mode_norm
        self.name_layer_conv = name_layer_conv
        self.num_channels_in = num_channels_in
        self.num_channels_hidden = num_channels_hidden
        self.nums_channels_hidden_projector_class = nums_channels_hidden_projector_class
        self.nums_channels_hidden_encoder_position = nums_channels_hidden_encoder_position
        self.nums_channels_hidden_head = nums_channels_hidden_head
        self.nums_channels_hidden_projector_language = nums_channels_hidden_projector_language
        self.nums_channels_hidden_referral = nums_channels_hidden_referral
        self.num_channels_out_projector_class = num_channels_out_projector_class
        self.num_channels_out_encoder_position = num_channels_out_encoder_position
        self.num_layers_referral = num_layers_referral
        self.num_samples = num_samples
        self.prob_dropout_projector_class = prob_dropout_projector_class
        self.prob_dropout_encoder_gnn = prob_dropout_encoder_gnn
        self.prob_dropout_encoder_points = prob_dropout_encoder_points
        self.prob_dropout_encoder_position = prob_dropout_encoder_position
        self.prob_dropout_head = prob_dropout_head
        self.prob_dropout_projector_language = prob_dropout_projector_language
        self.prob_dropout_referral = prob_dropout_referral
        self.projector_language = None
        self.scale_norm = scale_norm
        self.use_concat = use_concat
        self.use_norm_pair = use_norm_pair
        self.use_residual = use_residual
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.encoder_points = EncoderPointsPointNetPP(prob_dropout=self.prob_dropout_encoder_points)
        self.encoder_position = EncoderPositionMLP(
            num_channels_in=6,
            nums_channels_hidden=self.nums_channels_hidden_encoder_position,
            num_channels_out=self.num_channels_out_encoder_position,
            prob_dropout=self.prob_dropout_encoder_position,
        )
        self.projector_class = MLP(
            num_channels_in=300,
            nums_channels_hidden=self.nums_channels_hidden_projector_class,
            num_channels_out=self.num_channels_out_projector_class,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_projector_class,
        )
        self.encoder_fusion = MLP(
            num_channels_in=768 + self.num_channels_out_encoder_position + self.num_channels_out_projector_class,
            nums_channels_hidden=(self.num_channels_hidden, self.num_channels_hidden),
            num_channels_out=self.num_channels_hidden,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_encoder_position,
        )
        self.encoder_language = EncoderLanguageBERT(name_weights="bert-base-uncased", num_channels_out=768, num_layers=4, num_heads=12, type_vocab_size=2)
        self.projector_language = MLP(
            num_channels_in=768,
            nums_channels_hidden=self.nums_channels_hidden_projector_language,
            num_channels_out=self.num_channels_hidden,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout_projector_language,
        )

        if self.use_concat:
            self.referral = ModelReferral(
                num_channels_in=self.num_channels_hidden,
                nums_channels_hidden=self.nums_channels_hidden_referral,
                num_layers=self.num_layers_referral,
                prob_dropout=self.prob_dropout_referral,
                use_encoding_positions=self.use_encoding_positions,
            )
        else:
            self.referral = ModelReferral2(
                num_channels_in=self.num_channels_hidden,
                nums_channels_hidden=self.nums_channels_hidden_referral,
                num_layers=self.num_layers_referral,
                prob_dropout=self.prob_dropout_referral,
                use_encoding_positions=self.use_encoding_positions,
            )

        self.head_grounding = HeadGroundingMLP(num_channels_in=self.num_channels_hidden, nums_channels_hidden=self.nums_channels_hidden_head, prob_dropout=self.prob_dropout_head)

    def forward(self, input):
        input_points = input["features"][None, ...]
        output_features = self.encoder_points(input_points)
        output_features = output_features[0]

        input_position = torch.cat((input["pos"], input["size"]), dim=1)
        output_position = self.encoder_position(input_position)

        input_class = input["embeddings_class"]
        output_class = self.projector_class(input_class)
        output_class = torch.nn.functional.layer_norm(output_class, (self.num_channels_out_projector_class,))

        input_fusion = torch.cat((output_features, output_class, output_position), dim=1)
        output_fusion = self.encoder_fusion(input_fusion)
        output_fusion = torch.nn.functional.layer_norm(output_fusion, (self.num_channels_hidden,))

        # Have output_language of shape (B, MAX_NUM_TOKENS, C) and masks
        # Have output_fusion embeddings of shape (N, num_channels_HIDDEN) where N is the number of objects flattened over the batch and input["batch"] can be used for association

        # Regroup objects by batch
        list_embeddings_object = []
        list_positions = []
        for b in range(len(input["ptr"]) - 1):
            list_embeddings_object.append(output_fusion[input["batch"] == b])
            list_positions.append(input["pos"][input["batch"] == b])
        # Pad to [B, N_max, D]
        embeddings_object = torch.nn.utils.rnn.pad_sequence(list_embeddings_object, batch_first=True)
        positions = torch.nn.utils.rnn.pad_sequence(list_positions, batch_first=True)

        mask_object = torch.zeros(embeddings_object.size(0), embeddings_object.size(1), dtype=torch.bool, device=input["features"].device)
        for b, objs_b in enumerate(list_embeddings_object):
            mask_object[b, : objs_b.size(0)] = True

        input_language = dict(
            sentence=input["sentence"],
            device=input["features"].device,
        )
        output_language = self.encoder_language(input_language)

        output_language["embeddings"] = self.projector_language(output_language["embeddings"])
        output_language["embeddings"] = torch.nn.functional.layer_norm(output_language["embeddings"], (self.num_channels_hidden,))

        input_referral = dict(
            embeddings_object=embeddings_object,
            mask_object=mask_object,
            embeddings_language=output_language["embeddings"],
            mask_language=output_language["mask"],
            positions=positions,
        )
        output_referral = self.referral(input_referral)

        input_grounding = dict(embeddings=output_referral, mask=mask_object)
        output_logits = self.head_grounding(input_grounding)

        output = dict(
            logits=output_logits,
            mask=mask_object,
        )

        return output
