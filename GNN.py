import torch
from torch.nn import Linear, ReLU, Dropout, Sequential, ModuleDict, LazyLinear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, HeteroConv


class GNN(torch.nn.Module):
    """
    Heterogeneous GNN for university course timetabling, using the architecture
    from "Enhancing GAs with GNNs" (Cornei & Breaban, 2026) adapted to a
    course-scheduling graph with 5 node types:

        course    (6 features)  — central decision variable
        professor (3 features)  — teaches courses, involved in H2
        Room      (3 features)  — hosts courses, involved in H3/H4
        program   (4 features)  — groups of courses, involved in S2
        timeslot  (3 features)  — when courses happen, aggregates conflicts

    Edge types (made bidirectional by T.ToUndirected):
        course  --taught_by-->    professor
        course  --take_place_in-> Room
        course  --belong_to-->    program
        course  --when-->         timeslot
        + all reverses

    Architecture (from the paper's best config):
        - 4 heterogeneous TransformerConv layers, 8 attention heads
        - Concatenation of course embedding with its professor, room, program,
          and timeslot embeddings
        - 4-layer MLP outputting timeslot probabilities per course
    """

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
        self.nb_layers_conv = nb_layers_conv
        self.dropout_conv = dropout_conv
        self.dropout_mlp = dropout_mlp

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

        # --- MLP head ---
        # Concatenate course embedding with its professor, room, program, timeslot
        mlp_input_dim = hidden_channels * 5
        mlp_layers = []
        for i in range(nb_layers_mlp - 1):
            in_dim = mlp_input_dim if i == 0 else hidden_channels
            mlp_layers.append(Linear(in_dim, hidden_channels))
            mlp_layers.append(ReLU())
            if dropout_mlp > 0:
                mlp_layers.append(Dropout(dropout_mlp))
        mlp_layers.append(Linear(hidden_channels, num_timeslots))
        self.mlp = Sequential(*mlp_layers)

    def forward(self, x_dict, edge_index_dict, course_to_professor,
                course_to_room, course_to_program, course_to_timeslot):
        """
        Args:
            x_dict: node type -> feature tensor from HeteroData.x_dict
            edge_index_dict: edge type -> [2, E] from HeteroData.edge_index_dict
            course_to_professor: LongTensor [C] — professor index for each course
            course_to_room: LongTensor [C] — room index for each course
            course_to_program: LongTensor [C] — program index for each course
            course_to_timeslot: LongTensor [C] — timeslot index for each course

        Returns:
            Tensor [C, num_timeslots] — logits over timeslot assignments per course
        """
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
        course_emb = h_dict['course']                                # [C, hidden]
        prof_emb = h_dict['professor'][course_to_professor]          # [C, hidden]
        room_emb = h_dict['Room'][course_to_room]                    # [C, hidden]
        prog_emb = h_dict['program'][course_to_program]              # [C, hidden]
        ts_emb = h_dict['timeslot'][course_to_timeslot]              # [C, hidden]

        combined = torch.cat([course_emb, prof_emb, room_emb, prog_emb, ts_emb], dim=-1)

        return self.mlp(combined)  # [C, num_timeslots]

    def predict_timeslots(self, x_dict, edge_index_dict, course_to_professor,
                          course_to_room, course_to_program, course_to_timeslot):
        """Return argmax timeslot assignment for each course."""
        logits = self.forward(
            x_dict, edge_index_dict,
            course_to_professor, course_to_room,
            course_to_program, course_to_timeslot,
        )
        return logits.argmax(dim=-1)


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