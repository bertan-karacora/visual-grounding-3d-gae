import logging
import random

import numpy as np
import scipy as sp
import torch
import torch_geometric as torch_geo

from project.models.encoder_position import EncoderPositionMLP
from project.models.mlp import MLP
from project.models.modules import NormPair
from project.models.pointnetpp import EncoderPointsPointNetPP


_LOGGER = logging.getLogger(__name__)


def outer(a, b):
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


def match_hungarian(predictions, targets):
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)

    predictions, targets = outer(predictions, targets)
    error = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=1))

    # Detach, we only need matching
    costs = error.detach().cpu().numpy()
    costs[costs == np.inf] = 0.0
    indices = [sp.optimize.linear_sum_assignment(c) for c in costs]

    return indices, error


class EncoderGNN(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_channels_out, name_layer_conv="GCNConv", mode_norm="PN-SCS", scale_norm=20, prob_dropout=None, use_residual=False, use_norm_pair=False, mode_jk="max"):
        super().__init__()

        self.mode_norm = mode_norm
        self.mode_jk = mode_jk
        self.module_norm = None
        self.name_layer_conv = name_layer_conv
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.nums_channels_hidden = nums_channels_hidden
        self.prob_dropout = prob_dropout
        self.scale_norm = scale_norm
        self.use_norm_pair = use_norm_pair
        self.use_residual = use_residual

        self._init()

    def _init(self):
        nums_channels = [self.num_channels_in] + list(self.nums_channels_hidden) + [self.num_channels_out]

        if self.use_norm_pair:
            self.module_norm = NormPair(self.mode_norm, self.scale_norm)
        else:
            self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(c) for c in nums_channels[1:]])

        layers = []
        for num_channels_i, num_channels_o in zip(nums_channels[:-2], nums_channels[1:-1]):
            if self.name_layer_conv == "GATConv":
                layers.append(torch_geo.nn.GATConv(num_channels_i, num_channels_o, heads=4, concat=False))
            else:
                layer_conv = getattr(torch_geo.nn, self.name_layer_conv)
                layers.append(layer_conv(num_channels_i, num_channels_o))

        if self.name_layer_conv == "GATConv":
            layers.append(torch_geo.nn.GATConv(nums_channels[-2], nums_channels[-1], heads=4, concat=False))
        else:
            layer_conv = getattr(torch_geo.nn, self.name_layer_conv)
            layers.append(layer_conv(nums_channels[-2], nums_channels[-1]))

        self.layers = torch.nn.ModuleList(layers)

        self.jk = torch_geo.nn.JumpingKnowledge(self.mode_jk, channels=nums_channels[-1], num_layers=len(self.layers))

    def _forward(self, input):
        output = input["features"]

        outputs_intermediate = []
        for i, module in enumerate(self.layers[:-1]):
            output_prev = output
            output = module(output, input["edge_index"])
            if self.use_norm_pair:
                output = self.module_norm(output)
            else:
                output = self.norms[i](output)
            output = torch.nn.functional.relu(output)
            if self.prob_dropout is not None:
                output = torch.nn.functional.dropout(output, p=self.prob_dropout, training=self.training)

            if self.use_residual and output.shape == output_prev.shape:
                output = output + output_prev

            outputs_intermediate.append(output)

        # Final layer without activation
        output_embeddings = self.layers[-1](outputs_intermediate[-1], input["edge_index"])
        if self.use_norm_pair:
            output_embeddings = self.module_norm(output_embeddings)
        else:
            output_embeddings = self.norms[-1](output_embeddings)
        if self.use_residual and output_embeddings.shape == outputs_intermediate[-1].shape:
            output_embeddings = output_embeddings + outputs_intermediate[-1]

        output_embeddings = self.jk(outputs_intermediate + [output_embeddings])

        outputs_intermediate = list(reversed(outputs_intermediate))

        return output_embeddings, outputs_intermediate

    def forward(self, input):
        output_embeddings, outputs_intermediate = self._forward(input)

        return output_embeddings

    def forward_intermediates(self, input):
        output_embeddings, outputs_intermediate = self._forward(input)

        output = dict(
            embeddings=output_embeddings,
            embeddings_intermediate=outputs_intermediate,
        )
        return output


class DecoderMLP(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out_features_points, num_samples, num_decoder, num_channels_out_features_class=300, num_channels_out_position=6):
        super().__init__()

        self.num_channels_in = num_channels_in
        self.num_channels_out_features_class = num_channels_out_features_class
        self.num_channels_out_features_points = num_channels_out_features_points
        self.num_channels_out_position = num_channels_out_position
        self.num_decoder = num_decoder
        self.num_samples = num_samples

        self._init()

    def _init(self):
        self.decoder_degrees = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=(self.num_channels_in,) * self.num_decoder,
            num_channels_out=1,
            name_layer_norm="BatchNorm1d",
        )
        self.decoder_features_points = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=(self.num_channels_in,) * self.num_decoder,
            num_channels_out=self.num_channels_out_features_points,
            name_layer_norm="BatchNorm1d",
        )
        self.decoder_features_class = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=(self.num_channels_in,) * self.num_decoder,
            num_channels_out=self.num_channels_out_features_class,
            name_layer_norm="BatchNorm1d",
        )
        self.decoder_positions = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=(self.num_channels_in,) * self.num_decoder,
            num_channels_out=self.num_channels_out_position,
            name_layer_norm="BatchNorm1d",
        )

        self.decoders_layer = torch.nn.ModuleList(
            [
                MLP(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=(self.num_channels_in, self.num_channels_in, self.num_channels_in),
                    num_channels_out=self.num_channels_in,
                )
                for i in range(self.num_decoder)
            ]
        )

        self.mlp_mean = torch.nn.Linear(self.num_channels_in, self.num_channels_in)
        self.mlp_sigma = torch.nn.Linear(self.num_channels_in, self.num_channels_in)

        self.distribution = torch.distributions.Normal(torch.zeros(self.num_samples, self.num_channels_in), torch.ones(self.num_samples, self.num_channels_in))

    def decode_degrees(self, input):
        output = input["embeddings"]
        output = self.decoder_degrees(output)
        output = torch.nn.functional.relu(output)
        return output

    def decode_features_points(self, input):
        output = input["embeddings"]
        output = self.decoder_features_points(output)
        return output

    def decode_features_class(self, input):
        output = input["embeddings"]
        output = self.decoder_features_class(output)
        return output

    def decode_positions(self, input):
        output = input["embeddings"]
        output = self.decoder_positions(output)
        return output

    def sample_neighbors(self, idxs, dict_neighbors, embedding_dest):
        """Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding"""
        list_embeddings_sampled = []
        masks_len = []
        for idx in idxs:
            embeddings_sampled = []
            idxs_neighbors = dict_neighbors[idx]

            if len(idxs_neighbors) < self.num_samples:
                mask_len = len(idxs_neighbors)
                idxs_sample = idxs_neighbors
            else:
                idxs_sample = random.sample(idxs_neighbors, self.num_samples)
                mask_len = self.num_samples

            for idx_sample in idxs_sample:
                embeddings_sampled.append(embedding_dest[idx_sample].tolist())

            if len(embeddings_sampled) < self.num_samples:
                for _ in range(self.num_samples - len(embeddings_sampled)):
                    embeddings_sampled.append(torch.zeros(self.num_channels_in).tolist())

            list_embeddings_sampled.append(embeddings_sampled)
            masks_len.append(mask_len)

        return list_embeddings_sampled, masks_len

    def reconstruct_neighbors(self, module_generator, idxs_neighbors, neighbor_dict, embedding_origin, embedding_dest):
        list_embeddings_sampled, masks_len = self.sample_neighbors(idxs_neighbors, neighbor_dict, embedding_dest)

        for i, embeddings_neighbor in enumerate(list_embeddings_sampled):
            # Generating h^k_v, reparameterization trick
            idx = idxs_neighbors[i]
            mask_len1 = masks_len[i]

            mean = embedding_origin[idx].repeat(self.num_samples, 1)
            mean = self.mlp_mean(mean)

            sigma = embedding_origin[idx].repeat(self.num_samples, 1)
            sigma = self.mlp_sigma(sigma)

            std_z = self.distribution.sample()
            std_z = std_z.to(embedding_origin.device)
            var = mean + sigma.exp() * std_z
            nhij = module_generator(var)
            generated_neighbors = nhij

            generated_neighbors = generated_neighbors[None, ...]
            target_neighbors = torch.FloatTensor(embeddings_neighbor)[None, ...]
            target_neighbors = target_neighbors.to(embedding_origin.device)

            idxs_matching, errors = match_hungarian(generated_neighbors[:, :mask_len1, :], target_neighbors[:, :mask_len1, :])

        return idxs_matching, errors

    def decode_neighbors(self, input):
        neighbor_dict = {}
        for in_node, out_node in zip(input["edge_index"][0], input["edge_index"][1]):
            if in_node.item() not in neighbor_dict:
                neighbor_dict[in_node.item()] = []
            neighbor_dict[in_node.item()].append(out_node.item())

        # Sample multiple times to remove noise
        samples = []
        for _ in range(3):
            output_neighbor = []

            idxs = list(range(len(input["embeddings"])))
            embeddings = [input["embeddings"]] + input["embeddings_intermediate"]
            for i in range(len(embeddings) - 1):
                idxs, error = self.reconstruct_neighbors(self.decoders_layer[i], idxs, neighbor_dict, embeddings[i], embeddings[i + 1])
                output_neighbor.append((idxs, error))
                idxs = idxs[0][1]

            samples.append(output_neighbor)

        return samples

    def forward(self, input):
        output_degrees = self.decode_degrees(input)
        output_features_points = self.decode_features_points(input)
        output_features_class = self.decode_features_class(input)
        output_positions = self.decode_positions(input)
        output_neighbors = self.decode_neighbors(input)

        output = dict(
            degrees=output_degrees,
            features_points=output_features_points,
            features_class=output_features_class,
            positions=output_positions,
            neighbors=output_neighbors,
        )
        return output


class GAE(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden_gnn, num_channels_out, num_samples, name_layer_conv="GCNConv", mode_norm="PN-SCS", scale_norm=20, prob_dropout=None, use_residual=False, use_norm_pair=False, mode_jk="max"):
        super().__init__()

        self.decoder_mlp = None
        self.encoder_gnn = None
        self.mode_norm = mode_norm
        self.name_layer_conv = name_layer_conv
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.nums_channels_hidden_gnn = nums_channels_hidden_gnn
        self.num_samples = num_samples
        self.prob_dropout = prob_dropout
        self.scale_norm = scale_norm
        self.use_residual = use_residual
        self.use_norm_pair = use_norm_pair
        self.mode_jk = mode_jk

        self._init()

    def _init(self):
        self.encoder_gnn = EncoderGNN(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=self.nums_channels_hidden_gnn,
            num_channels_out=self.num_channels_out,
            name_layer_conv=self.name_layer_conv,
            mode_norm=self.mode_norm,
            scale_norm=self.scale_norm,
            prob_dropout=self.prob_dropout,
            use_residual=self.use_residual,
            use_norm_pair=self.use_norm_pair,
            mode_jk=self.mode_jk,
        )
        self.decoder_mlp = DecoderMLP(self.num_channels_out, self.num_channels_in, self.num_samples, num_decoder=len(self.nums_channels_hidden_gnn))

    def encode(self, input):
        output = self.encoder_gnn(input)
        return output

    def decode(self, input):
        output = self.decoder_mlp(input)
        return output

    def forward(self, input):
        output_encoder = self.encoder_gnn.forward_intermediates(input)

        input_decoder = dict(
            embeddings=output_encoder["embeddings"],
            embeddings_intermediate=output_encoder["embeddings_intermediate"],
            edge_index=input["edge_index"],
        )
        output_decoder = self.decoder_mlp(input_decoder)

        output = dict(
            embeddings=output_encoder["embeddings"],
            embeddings_intermediate=output_encoder["embeddings_intermediate"],
            degrees=output_decoder["degrees"],
            features_points=output_decoder["features_points"],
            features_class=output_decoder["features_class"],
            positions=output_decoder["positions"],
            neighbors=output_decoder["neighbors"],
        )
        return output


class GAEScan(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_hidden,
        nums_channels_hidden_encoder_position,
        num_channels_out_encoder_position,
        nums_channels_hidden_projector_class,
        num_channels_out_projector_class,
        nums_channels_hidden_encoder_gnn,
        num_samples,
        name_layer_conv="GCNConv",
        mode_norm="PN-SCS",
        scale_norm=20,
        prob_dropout_encoder_points=None,
        prob_dropout_projector_class=None,
        prob_dropout_encoder_gnn=None,
        prob_dropout_encoder_position=None,
        use_residual=False,
        use_norm_pair=False,
        mode_jk="max",
    ):
        super().__init__()

        self.encoder_language = None
        self.encoder_points = None
        self.encoder_position = None
        self.mode_jk = mode_jk
        self.mode_norm = mode_norm
        self.name_layer_conv = name_layer_conv
        self.num_channels_in = num_channels_in
        self.num_channels_hidden = num_channels_hidden
        self.nums_channels_hidden_projector_class = nums_channels_hidden_projector_class
        self.nums_channels_hidden_encoder_gnn = nums_channels_hidden_encoder_gnn
        self.nums_channels_hidden_encoder_position = nums_channels_hidden_encoder_position
        self.num_channels_out_projector_class = num_channels_out_projector_class
        self.num_channels_out_encoder_position = num_channels_out_encoder_position
        self.num_samples = num_samples
        self.prob_dropout_projector_class = prob_dropout_projector_class
        self.prob_dropout_encoder_gnn = prob_dropout_encoder_gnn
        self.prob_dropout_encoder_points = prob_dropout_encoder_points
        self.prob_dropout_encoder_position = prob_dropout_encoder_position
        self.scale_norm = scale_norm
        self.use_norm_pair = use_norm_pair
        self.use_residual = use_residual

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
        self.gae = GAE(
            num_channels_in=self.num_channels_out_projector_class,
            nums_channels_hidden_gnn=self.nums_channels_hidden_encoder_gnn,
            num_channels_out=self.num_channels_hidden,
            num_samples=self.num_samples,
            name_layer_conv=self.name_layer_conv,
            mode_norm=self.mode_norm,
            scale_norm=self.scale_norm,
            prob_dropout=self.prob_dropout_encoder_gnn,
            use_residual=self.use_residual,
            mode_jk=self.mode_jk,
            use_norm_pair=self.use_norm_pair,
        )

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

        input_gae = dict(
            features=output_fusion,
            edge_index=input["edge_index"],
        )
        output_gae = self.gae(input_gae)

        output = dict(
            features_points_raw=output_features,
            embeddings=output_gae["embeddings"],
            degrees=output_gae["degrees"],
            features_points=output_gae["features_points"],
            features_class=output_gae["features_class"],
            positions=output_gae["positions"],
            neighbors=output_gae["neighbors"],
        )

        return output
