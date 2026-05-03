import torch
from torch.nn import Linear, ReLU, Dropout, Sequential, ModuleDict, LazyLinear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, HeteroConv


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_timeslots: int = 25,
        hidden_channels: int = 64,
        course_feature_dim: int = 12,
        professor_feature_dim: int = 3,
        room_feature_dim: int = 5,
        program_feature_dim: int = 4,
        timeslot_feature_dim: int = 5,
        nb_layers_conv: int = 4,
        nb_layers_mlp: int = 4,
        nb_heads: int = 8,
        dropout_conv: float = 0.0,
        dropout_mlp: float = 0.0,
    ):
        super().__init__()

        self.num_timeslots = num_timeslots
        self.hidden_channels = hidden_channels

        # --- Input projections ---
        self.input_projs = ModuleDict({
            'course': Linear(course_feature_dim, hidden_channels),
            'professor': Linear(professor_feature_dim, hidden_channels),
            'Room': Linear(room_feature_dim, hidden_channels),
            'program': Linear(program_feature_dim, hidden_channels),
            'timeslot': Linear(timeslot_feature_dim, hidden_channels),
        })

        # --- Heterogeneous TransformerConv layers ---
        # 8 edge types after T.ToUndirected()
        self.conv_layers = ModuleList()
        for _ in range(nb_layers_conv):
            conv_dict = {
                ('course', 'taught_by', 'professor'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('course', 'take_place_in', 'Room'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('course', 'belong_to', 'program'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('course', 'when', 'timeslot'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('professor', 'rev_taught_by', 'course'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('Room', 'rev_take_place_in', 'course'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('program', 'rev_belong_to', 'course'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
                ('timeslot', 'rev_when', 'course'): TransformerConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // nb_heads,
                    heads=nb_heads, dropout=dropout_conv,
                ),
            }
            self.conv_layers.append(HeteroConv(conv_dict, aggr='sum'))

        # --- Shared feature concatenation dimension ---
        mlp_input_dim = hidden_channels * 5  # course + prof + room + program + timeslot

        # --- Timeslot prediction head ---
        ts_layers = []
        for i in range(nb_layers_mlp - 1):
            in_dim = mlp_input_dim if i == 0 else hidden_channels
            ts_layers.append(Linear(in_dim, hidden_channels))
            ts_layers.append(ReLU())
            if dropout_mlp > 0:
                ts_layers.append(Dropout(dropout_mlp))
        ts_layers.append(Linear(hidden_channels, num_timeslots))
        self.timeslot_head = Sequential(*ts_layers)

        # --- Room prediction head (pointer network) ---
        # Instead of a fixed-size classification head, we project the course
        # context to a query vector and compute a dot-product score against
        # every room embedding in the instance. This naturally handles
        # variable numbers of rooms across instances and batches.
        room_query_layers = []
        for i in range(nb_layers_mlp - 1):
            in_dim = mlp_input_dim if i == 0 else hidden_channels
            room_query_layers.append(Linear(in_dim, hidden_channels))
            room_query_layers.append(ReLU())
            if dropout_mlp > 0:
                room_query_layers.append(Dropout(dropout_mlp))
        room_query_layers.append(Linear(hidden_channels, hidden_channels))
        self.room_query_head = Sequential(*room_query_layers)

        # Learnable projection for room keys (from backbone room embeddings)
        self.room_key_proj = Linear(hidden_channels, hidden_channels)

    def _run_backbone(self, x_dict, edge_index_dict, course_to_professor,
                      course_to_room, course_to_program, course_to_timeslot):
        """Run the shared backbone and return h_dict and combined course context."""
        # Project all node types to hidden dimension
        h_dict = {
            ntype: self.input_projs[ntype](x_dict[ntype])
            for ntype in x_dict
        }

        # Apply heterogeneous TransformerConv layers with residual connections
        for conv in self.conv_layers:
            h_dict_new = conv(h_dict, edge_index_dict)
            h_dict = {
                key: F.relu(h_dict_new[key] + h_dict[key])
                for key in h_dict
            }

        # Concatenate course embedding with related node embeddings
        course_emb = h_dict['course']
        prof_emb = h_dict['professor'][course_to_professor]
        room_emb = h_dict['Room'][course_to_room]
        prog_emb = h_dict['program'][course_to_program]
        ts_emb = h_dict['timeslot'][course_to_timeslot]

        combined = torch.cat([course_emb, prof_emb, room_emb, prog_emb, ts_emb], dim=-1)

        return h_dict, combined

    def forward(self, x_dict, edge_index_dict, course_to_professor,
                course_to_room, course_to_program, course_to_timeslot,
                course_batch=None, room_batch=None):
        """
        Returns:
            ts_logits:   Tensor [C, num_timeslots] — logits over timeslot assignments
            room_logits: Tensor [C, R_max] — logits over room assignments, where
                         R_max is the max number of rooms in any instance in the
                         batch. Positions for rooms not belonging to a course's
                         instance are masked to -inf.
        """
        h_dict, combined = self._run_backbone(
            x_dict, edge_index_dict,
            course_to_professor, course_to_room,
            course_to_program, course_to_timeslot,
        )

        # --- Timeslot head ---
        ts_logits = self.timeslot_head(combined)

        # --- Room pointer head ---
        all_room_embs = h_dict['Room']           # [R_total, H]
        room_keys = self.room_key_proj(all_room_embs)  # [R_total, H]
        course_queries = self.room_query_head(combined) # [C, H]

        R_total = all_room_embs.size(0)
        C = course_queries.size(0)

        if course_batch is None and room_batch is None:
            # Single graph (no batching): simple dot-product scores
            # course_queries: [C, H], room_keys: [R, H] => [C, R]
            room_logits = torch.matmul(course_queries, room_keys.t())
        else:
            # Batched: each course should only score rooms from its own graph.
            # Build a padded [C, R_max] logit matrix with -inf masking.
            assert course_batch is not None and room_batch is not None, \
                "Both course_batch and room_batch are needed for batched forward"

            num_graphs = course_batch.max().item() + 1

            # Count rooms per graph and find R_max
            rooms_per_graph = torch.zeros(num_graphs, dtype=torch.long,
                                          device=room_keys.device)
            rooms_per_graph.scatter_add_(0, room_batch,
                                         torch.ones(R_total, dtype=torch.long,
                                                    device=room_keys.device))
            R_max = rooms_per_graph.max().item()

            # Build local room index within each graph
            # room_local_idx[i] = position of room i among rooms in its graph
            room_local_idx = torch.zeros(R_total, dtype=torch.long,
                                         device=room_keys.device)
            room_offset = torch.zeros(num_graphs, dtype=torch.long,
                                      device=room_keys.device)
            for i in range(R_total):
                g = room_batch[i].item()
                room_local_idx[i] = room_offset[g]
                room_offset[g] += 1

            # Pad room keys into [num_graphs, R_max, H]
            padded_keys = torch.zeros(num_graphs, R_max, self.hidden_channels,
                                      device=room_keys.device)
            padded_keys[room_batch, room_local_idx] = room_keys

            # Build mask: [num_graphs, R_max], True where valid
            valid_mask = torch.zeros(num_graphs, R_max, dtype=torch.bool,
                                     device=room_keys.device)
            valid_mask[room_batch, room_local_idx] = True

            # Gather the padded keys and mask for each course's graph
            # course_graph_keys: [C, R_max, H]
            course_graph_keys = padded_keys[course_batch]
            # course_valid_mask: [C, R_max]
            course_valid_mask = valid_mask[course_batch]

            # Dot product: [C, R_max]
            room_logits = torch.bmm(
                course_graph_keys,
                course_queries.unsqueeze(-1)
            ).squeeze(-1)

            # Mask out rooms from other graphs
            room_logits[~course_valid_mask] = float('-inf')

        return ts_logits, room_logits

    def predict(self, x_dict, edge_index_dict, course_to_professor,
                course_to_room, course_to_program, course_to_timeslot,
                course_batch=None, room_batch=None):
        """Return argmax timeslot and room assignment for each course."""
        ts_logits, room_logits = self.forward(
            x_dict, edge_index_dict,
            course_to_professor, course_to_room,
            course_to_program, course_to_timeslot,
            course_batch, room_batch,
        )
        return ts_logits.argmax(dim=-1), room_logits.argmax(dim=-1)


def extract_mapping_tensors(sched, ds):
    """
    Build the course -> {professor, room, program, timeslot} index mappings
    needed by the forward pass, from the schedule DataFrame and Instance.

    Args:
        sched: DataFrame with columns Index, P-Index, Room, dl, timeslot
        ds: Instance object

    Returns:
        course_to_professor, course_to_room, course_to_program, course_to_timeslot
        (all LongTensors of shape [C])
    """
    course_to_idx = {c: i for i, c in enumerate(ds.course_list)}
    prof_to_idx = {p: i for i, p in enumerate(ds.professors['P-Index'].drop_duplicates())}
    room_to_idx = {r: i for i, r in enumerate(ds.rooms['Host Key'].drop_duplicates())}
    program_to_idx = {g: i for i, g in enumerate(sched['dl'].unique().tolist())}
    timeslot_to_idx = {t: i for i, t in enumerate(list(ds.timeslots))}

    C = len(ds.course_list)
    c2prof = torch.zeros(C, dtype=torch.long)
    c2room = torch.zeros(C, dtype=torch.long)
    c2prog = torch.zeros(C, dtype=torch.long)
    c2ts = torch.zeros(C, dtype=torch.long)

    for _, row in sched.iterrows():
        ci = course_to_idx[row['Index']]
        c2prof[ci] = prof_to_idx[row['P-Index']]
        c2room[ci] = room_to_idx[row['Room']]
        c2prog[ci] = program_to_idx[row['dl']]
        c2ts[ci] = timeslot_to_idx[row['timeslot']]

    return c2prof, c2room, c2prog, c2ts

if __name__ == '__main__':
    model = GNN(
        num_timeslots=25,
        hidden_channels=64,
        nb_layers_conv=4,
        nb_layers_mlp=4,
        nb_heads=8,
        dropout_conv=0.0,
        dropout_mlp=0.0,
    )

    print(model)
    # print(f"\nEdge types: {data.edge_types}")

    # Forward pass
    # with torch.no_grad():
    #     logits = model(data.x_dict, data.edge_index_dict, c2prof, c2room, c2prog, c2ts)
    #     print(f"\nOutput shape: {logits.shape}")  # [50, 25]
    #
    #     predicted = model.predict_timeslots(
    #         data.x_dict, data.edge_index_dict, c2prof, c2room, c2prog, c2ts
    #     )
    #     print(f"Predicted timeslots shape: {predicted.shape}")  # [50]
    #
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")